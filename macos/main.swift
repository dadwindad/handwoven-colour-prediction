// macOS wrapper for the web app in ../app: one window with a WKWebView.
//
// The web files are copied into the bundle (Contents/Resources/web) by build.sh and
// served through a custom URL scheme, weave://app/, so fetch("data/dataset.json")
// works (WebKit blocks fetch over file://) and localStorage gets a stable origin.
// Links to other sites open in the default browser.
import AppKit
import UniformTypeIdentifiers
import WebKit

let scheme = "weave"

final class BundleSchemeHandler: NSObject, WKURLSchemeHandler {
    let root: URL

    init(root: URL) { self.root = root.standardizedFileURL }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else { return }
        var path = url.path
        if path.isEmpty || path == "/" { path = "/index.html" }
        let file = root.appendingPathComponent(String(path.dropFirst())).standardizedFileURL
        // never serve anything outside the web folder
        guard file.path.hasPrefix(root.path + "/"), let data = try? Data(contentsOf: file) else {
            task.didReceive(HTTPURLResponse(url: url, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: nil)!)
            task.didFinish()
            return
        }
        let mime = UTType(filenameExtension: file.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
        let type = file.pathExtension == "webmanifest" ? "application/manifest+json" : mime
        let headers = ["Content-Type": type, "Content-Length": String(data.count), "Cache-Control": "no-cache"]
        task.didReceive(HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!)
        task.didReceive(data)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    let home = URL(string: "\(scheme)://app/index.html#look")!

    func applicationDidFinishLaunching(_ note: Notification) {
        let web = Bundle.main.resourceURL!.appendingPathComponent("web")
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(BundleSchemeHandler(root: web), forURLScheme: scheme)
        config.websiteDataStore = .default()

        webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.allowsMagnification = true

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1100, height: 800),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String ?? "สีผ้าทอโทเร"
        window.minSize = NSSize(width: 380, height: 560)
        window.contentView = webView
        window.setFrameAutosaveName("MainWindow")
        if !window.setFrameUsingName("MainWindow") { window.center() }
        window.makeKeyAndOrderFront(nil)

        buildMenu()
        let start = CommandLine.arguments.dropFirst().first(where: { $0.hasPrefix("#") })
        webView.load(URLRequest(url: start.flatMap { URL(string: "\(scheme)://app/index.html\($0)") } ?? home))
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    // Pages of the app stay in the window; everything else goes to the default browser.
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let url = action.request.url, url.scheme != scheme, url.scheme != "about" {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
        } else {
            decisionHandler(.allow)
        }
    }

    // target="_blank" links
    func webView(_ webView: WKWebView, createWebViewWith config: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url { NSWorkspace.shared.open(url) }
        return nil
    }

    // Self-check for build verification: WEAVE_SNAPSHOT=/tmp/x.png <executable> [#view]
    // waits for the page, prints what it rendered, saves a PNG of the window and quits.
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard let out = ProcessInfo.processInfo.environment["WEAVE_SNAPSHOT"] else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            let probe = "JSON.stringify({view: location.hash, loading: !!document.querySelector('.loading'),"
                + " notice: document.querySelector('.notice')?.textContent ?? null,"
                + " cells: document.querySelectorAll('#view *').length, mode: document.documentElement.dataset.mode})"
            webView.evaluateJavaScript(probe) { result, error in
                print("selftest:", result ?? "nil", error.map { "\($0)" } ?? "")
                webView.takeSnapshot(with: nil) { image, _ in
                    if let tiff = image?.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                       let png = rep.representation(using: .png, properties: [:]) {
                        try? png.write(to: URL(fileURLWithPath: out))
                    }
                    NSApp.terminate(nil)
                }
            }
        }
    }

    @objc func goHome() { webView.load(URLRequest(url: home)) }
    @objc func reload() { webView.reload() }
    @objc func setWeaverMode() { setMode("weaver") }
    @objc func setResearchMode() { setMode("research") }
    @objc func zoomIn() { webView.pageZoom = min(webView.pageZoom + 0.1, 3) }
    @objc func zoomOut() { webView.pageZoom = max(webView.pageZoom - 0.1, 0.5) }
    @objc func zoomReset() { webView.pageZoom = 1 }

    func setMode(_ mode: String) {
        webView.evaluateJavaScript("document.querySelector('.mode-switch button[data-mode=\"\(mode)\"]')?.click()")
    }

    func buildMenu() {
        let main = NSMenu()
        let name = window.title

        let app = NSMenu()
        app.addItem(withTitle: "เกี่ยวกับ \(name)", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "ซ่อน \(name)", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(.separator())
        app.addItem(withTitle: "ออกจาก \(name)", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let edit = NSMenu(title: "แก้ไข")
        edit.addItem(withTitle: "เลิกทำ", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "ทำซ้ำ", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "ตัด", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "คัดลอก", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "วาง", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "เลือกทั้งหมด", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        let view = NSMenu(title: "มุมมอง")
        view.addItem(withTitle: "หน้าแรก", action: #selector(goHome), keyEquivalent: "1")
        view.addItem(withTitle: "โหลดใหม่", action: #selector(reload), keyEquivalent: "r")
        view.addItem(.separator())
        view.addItem(withTitle: "โหมดช่างทอ", action: #selector(setWeaverMode), keyEquivalent: "")
        view.addItem(withTitle: "โหมดนักวิจัย", action: #selector(setResearchMode), keyEquivalent: "")
        view.addItem(.separator())
        view.addItem(withTitle: "ขยาย", action: #selector(zoomIn), keyEquivalent: "+")
        view.addItem(withTitle: "ย่อ", action: #selector(zoomOut), keyEquivalent: "-")
        view.addItem(withTitle: "ขนาดจริง", action: #selector(zoomReset), keyEquivalent: "0")
        view.addItem(.separator())
        view.addItem(withTitle: "เต็มจอ", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
            .keyEquivalentModifierMask = [.command, .control]

        let win = NSMenu(title: "หน้าต่าง")
        win.addItem(withTitle: "ย่อเก็บ", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "ปิด", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")

        for (title, menu) in [(name, app), ("แก้ไข", edit), ("มุมมอง", view), ("หน้าต่าง", win)] {
            let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
            item.submenu = menu
            main.addItem(item)
        }
        for item in view.items where item.action != nil && item.action != #selector(NSWindow.toggleFullScreen(_:)) {
            item.target = self
        }
        NSApp.mainMenu = main
        NSApp.windowsMenu = win
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
