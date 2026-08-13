import CoreGraphics
import CoreImage
import Foundation
import Vision

struct FaceCandidate {
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let confidence: Double
    let score: Double
    let pass: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func clamp(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
    min(max(value, lower), upper)
}

func intersectionOverUnion(_ a: FaceCandidate, _ b: FaceCandidate) -> Double {
    let x1 = max(a.x, b.x)
    let y1 = max(a.y, b.y)
    let x2 = min(a.x + a.w, b.x + b.w)
    let y2 = min(a.y + a.h, b.y + b.h)
    let intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    let union = a.w * a.h + b.w * b.h - intersection
    return union > 0 ? intersection / union : 0
}

func suppressDuplicates(_ faces: [FaceCandidate], threshold: Double = 0.28) -> [FaceCandidate] {
    var kept: [FaceCandidate] = []
    let sorted = faces.sorted { left, right in
        if left.score == right.score {
            return left.confidence > right.confidence
        }
        return left.score > right.score
    }

    for face in sorted {
        if kept.allSatisfy({ intersectionOverUnion(face, $0) < threshold }) {
            kept.append(face)
        }
    }

    return kept
}

func detectFaces(
    in image: CIImage,
    crop: CGRect,
    scale: CGFloat,
    pass: String,
    originalWidth: Double,
    originalHeight: Double
) -> [FaceCandidate] {
    let cropped = image
        .cropped(to: crop)
        .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))

    let request = VNDetectFaceRectanglesRequest()
    request.revision = VNDetectFaceRectanglesRequestRevision3
    let handler = VNImageRequestHandler(ciImage: cropped, options: [:])

    do {
        try handler.perform([request])
    } catch {
        return []
    }

    let cropWidth = Double(crop.width)
    let cropHeight = Double(crop.height)

    return (request.results ?? []).compactMap { observation in
        let box = observation.boundingBox
        let w = Double(box.width) * cropWidth
        let h = Double(box.height) * cropHeight
        if w < 8 || h < 8 {
            return nil
        }

        let x = Double(crop.minX) + Double(box.origin.x) * cropWidth
        let y = Double(crop.minY) + (1.0 - Double(box.origin.y) - Double(box.height)) * cropHeight
        let confidence = Double(observation.confidence)
        let sizeBoost = min(1.0, max(w, h) / 80.0)

        return FaceCandidate(
            x: clamp(x, 0, originalWidth),
            y: clamp(y, 0, originalHeight),
            w: clamp(w, 1, originalWidth),
            h: clamp(h, 1, originalHeight),
            confidence: confidence,
            score: confidence + sizeBoost * 0.08,
            pass: pass
        )
    }
}

func tileRects(width: Double, height: Double) -> [CGRect] {
    let maxSide = max(width, height)
    let tileSize: Double

    if maxSide >= 5000 {
        tileSize = 1100
    } else if maxSide >= 3000 {
        tileSize = 950
    } else {
        tileSize = 800
    }

    let overlap = tileSize * 0.28
    let step = max(320, tileSize - overlap)
    var rects: [CGRect] = []
    var y = 0.0

    while y < height {
        var x = 0.0
        let h = min(tileSize, height - y)
        let y0 = max(0, min(y, height - h))

        while x < width {
            let w = min(tileSize, width - x)
            let x0 = max(0, min(x, width - w))
            rects.append(CGRect(x: x0, y: y0, width: w, height: h))

            if x + tileSize >= width {
                break
            }
            x += step
        }

        if y + tileSize >= height {
            break
        }
        y += step
    }

    return rects
}

guard CommandLine.arguments.count == 2 else {
    fail("Usage: detect_faces <image-path>")
}

let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let image = CIImage(contentsOf: imageURL) else {
    fail("Could not load image")
}

let extent = image.extent
let width = Double(extent.width)
let height = Double(extent.height)
guard width > 0, height > 0 else {
    fail("Invalid image size")
}

let fullRect = CGRect(x: 0, y: 0, width: width, height: height)
let fullScale = CGFloat(min(2.0, max(1.0, 1800.0 / max(width, height))))
var candidates = detectFaces(
    in: image,
    crop: fullRect,
    scale: fullScale,
    pass: "full",
    originalWidth: width,
    originalHeight: height
)

for tile in tileRects(width: width, height: height) {
    let maxTileSide = max(Double(tile.width), Double(tile.height))
    let tileScale = CGFloat(min(4.0, max(1.4, 1500.0 / maxTileSide)))
    candidates += detectFaces(
        in: image,
        crop: tile,
        scale: tileScale,
        pass: "tile",
        originalWidth: width,
        originalHeight: height
    )
}

let faces = suppressDuplicates(candidates)
    .sorted { $0.x < $1.x }
    .map { face -> [String: Any] in
        [
            "x": face.x,
            "y": face.y,
            "w": face.w,
            "h": face.h,
            "confidence": face.confidence,
            "pass": face.pass,
        ]
    }

do {
    let payload: [String: Any] = [
        "faces": faces,
        "passes": [
            "full": 1,
            "tiles": tileRects(width: width, height: height).count,
        ],
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [])
    FileHandle.standardOutput.write(data)
} catch {
    fail("Could not encode JSON")
}
