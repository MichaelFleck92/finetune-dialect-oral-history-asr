import Foundation
import Vision
import AppKit

// Zeilenweise OCR mit dem macOS-Vision-Framework.
//   vision_ocr <bild> [--no-correction]
// Ausgabe je Zeile: x0 y0 x1 y1 (Pixel, Ursprung oben links) <TAB> Konfidenz <TAB> Text
// --no-correction schaltet die Sprachkorrektur ab. Der Vergleich beider Laeufe
// zeigt, wo die Korrektur ein Wort "verbessert" hat - dort ist die Lesung unsicher.
let args = CommandLine.arguments
guard args.count > 1, let img = NSImage(contentsOfFile: args[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("Bild nicht lesbar\n".data(using: .utf8)!); exit(1)
}
let W = CGFloat(cg.width), H = CGFloat(cg.height)
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = !args.contains("--no-correction")
req.recognitionLanguages = ["de-DE"]
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do { try handler.perform([req]) } catch { exit(2) }
for obs in (req.results ?? []) {
    guard let top = obs.topCandidates(1).first else { continue }
    let b = obs.boundingBox                      // normiert, Ursprung unten links
    let x0 = b.minX * W, x1 = b.maxX * W
    let y0 = (1 - b.maxY) * H, y1 = (1 - b.minY) * H
    print(String(format: "%.0f %.0f %.0f %.0f\t%.3f\t%@", x0, y0, x1, y1, top.confidence, top.string))
}
