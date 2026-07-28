import TensorFlowLite
import UIKit

final class MammogramClassifier {
    private var interpreter: Interpreter
    private let inputSize = 224   // matches INPUT_SIZE in the notebook

    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "second_look", ofType: "tflite") else {
            throw NSError(domain: "Model", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "second_look.tflite not found in bundle"])
        }
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }

    struct Result {
        let probability: Float   // P(WORTH_SECOND_LOOK)
        let tier: String         // "Low" / "Moderate" / "Elevated"
    }

    func classify(_ image: UIImage) throws -> Result {
        guard let inputData = MammogramPreprocessor.preprocess(image, size: Int32(inputSize)) else {
            throw NSError(domain: "Model", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Preprocessing failed"])
        }
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()
        let output = try interpreter.output(at: 0)
        let prob = output.data.toFloatArray().first ?? 0
        return Result(probability: prob, tier: Self.tier(for: prob))
    }

    // Tier cut-points from the team's confidence_to_tier() — provisional, not calibrated
    static func tier(for prob: Float) -> String {
        if prob < 0.33 { return "Low" }
        if prob < 0.66 { return "Moderate" }
        return "Elevated"
    }
}

// MARK: - Preprocessing helpers

private extension UIImage {
    /// Resize to size×size, convert to grayscale, normalize to [0,1] float32.
    /// Output layout: [1, size, size, 1] — single channel to match the model.
    func grayscaleFloatData(size: Int) -> Data? {
        guard let cgImage = cgImage else { return nil }

        // Draw into a 1-byte-per-pixel grayscale context
        var pixels = [UInt8](repeating: 0, count: size * size)
        guard let context = CGContext(
            data: &pixels, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))

        var floats = [Float](repeating: 0, count: size * size)
        for i in 0..<(size * size) {
            floats[i] = Float(pixels[i]) / 255.0
        }
        return floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

private extension Data {
    func toFloatArray() -> [Float] {
        withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
