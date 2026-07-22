//
//  ContentView.swift
//  SecondLook
//

import SwiftUI

// MARK: - Flow State

enum AppStage {
    case disclaimer
    case capture
    case scanning
    case results
}

// MARK: - Root View

struct ContentView: View {
    @State private var stage: AppStage = .disclaimer
    @State private var selectedImage: UIImage?
    @State private var result: MammogramClassifier.Result?

    var body: some View {
        ZStack {
            // Calm background for the whole app
            LinearGradient(
                colors: [Color(.systemBackground), Color(.systemGray6)],
                startPoint: .top, endPoint: .bottom
            )
            .ignoresSafeArea()

            switch stage {
            case .disclaimer:
                DisclaimerGateView {
                    withAnimation(.easeInOut) { stage = .capture }
                }
            case .capture:
                CaptureView { image in
                    selectedImage = image
                    withAnimation(.easeInOut) { stage = .scanning }
                }
            case .scanning:
                ScanningView(image: selectedImage) { classificationResult in
                    result = classificationResult
                    withAnimation(.easeInOut) { stage = .results }
                }
            case .results:
                ResultsView(image: selectedImage, result: result) {
                    // Back to capture — disclaimer stays accepted for this session
                    selectedImage = nil
                    result = nil
                    withAnimation(.easeInOut) { stage = .capture }
                }
            }
        }
    }
}

// MARK: - 1. Disclaimer Gate

struct DisclaimerGateView: View {
    let onAccept: () -> Void
    @State private var hasAgreed = false

    var body: some View {
        VStack {
            Spacer()

            // Centered content
            VStack(spacing: 24) {
                ZStack {
                    Circle()
                        .fill(Color.accentColor.opacity(0.12))
                        .frame(width: 140, height: 140)
                    Image(systemName: "cross.case.fill")
                        .font(.system(size: 56))
                        .foregroundStyle(.tint)
                }

                Text("Second Look")
                    .font(.largeTitle.bold())

                Text("This app provides a non-diagnostic, informational review of mammogram images. It does not replace professional medical evaluation. All analysis runs on-device — no images are stored or transmitted.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            }

            Spacer()

            // Bottom controls
            VStack(spacing: 20) {
                Toggle(isOn: $hasAgreed) {
                    Text("I understand this is not a medical diagnosis, and I will consult a healthcare professional for any concerns.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color(.secondarySystemBackground))
                )

                Button {
                    onAccept()
                } label: {
                    Text("Continue")
                        .frame(maxWidth: .infinity)
                        .padding(6)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!hasAgreed)
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - 2. Upload / Capture

struct CaptureView: View {
    let onImageSelected: (UIImage) -> Void
    @State private var showingPicker = false

    var body: some View {
        VStack(spacing: 20) {
            Spacer()

            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.12))
                    .frame(width: 140, height: 140)
                Image(systemName: "photo.badge.plus")
                    .font(.system(size: 56))
                    .foregroundStyle(.tint)
            }

            Text("Add an Image")
                .font(.title2.bold())

            Text("Choose a mammogram image from your library, or take a photo.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button {
                showingPicker = true
            } label: {
                Label("Choose Image", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
                    .padding(6)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 40)

            Spacer()
        }
        .sheet(isPresented: $showingPicker) {
            ImagePicker { image in
                onImageSelected(image)
            }
        }
    }
}

// MARK: - 3. Scanning

struct ScanningView: View {
    let image: UIImage?
    let onComplete: (MammogramClassifier.Result?) -> Void
    @State private var pulse = false

    var body: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.12))
                    .frame(width: 120, height: 120)
                    .scaleEffect(pulse ? 1.15 : 0.9)
                    .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true), value: pulse)
                Image(systemName: "waveform.and.magnifyingglass")
                    .font(.system(size: 44))
                    .foregroundStyle(.tint)
            }
            Text("Analyzing on-device…")
                .font(.headline)
            Label("Nothing leaves your phone", systemImage: "lock.fill")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .onAppear {
            pulse = true
            DispatchQueue.global(qos: .userInitiated).async {
                let result: MammogramClassifier.Result?
                do {
                    let classifier = try MammogramClassifier()
                    result = try classifier.classify(image ?? UIImage())
                } catch {
                    print("Inference failed: \(error)")
                    result = nil
                }
                DispatchQueue.main.async { onComplete(result) }
            }
        }
    }
}

// MARK: - 4. Results

struct ResultsView: View {
    let image: UIImage?
    let result: MammogramClassifier.Result?
    let onDone: () -> Void

    private var tierTitle: String {
        guard let result else { return "Analysis Unavailable" }
        switch result.tier {
        case "Low":      return "Low Concern"
        case "Moderate": return "Moderate — Worth a Second Look"
        default:         return "Elevated — Worth a Second Look"
        }
    }

    private var tierColor: Color {
        guard let result else { return .gray }
        switch result.tier {
        case "Low":      return .green
        case "Moderate": return .orange
        default:         return .red
        }
    }

    private var tierIcon: String {
        guard let result else { return "questionmark.circle.fill" }
        switch result.tier {
        case "Low":      return "checkmark.shield.fill"
        case "Moderate": return "exclamationmark.shield.fill"
        default:         return "exclamationmark.triangle.fill"
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .shadow(color: .black.opacity(0.15), radius: 10, y: 4)
                    .padding(.top)
            }

            // Concern tier badge (with confidence inside)
            HStack(spacing: 10) {
                Image(systemName: tierIcon)
                    .font(.title2)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Concern Tier")
                        .font(.caption)
                        .opacity(0.8)
                    Text(tierTitle)
                        .font(.title3.bold())
                }
                Spacer()
                if let result {
                    Text("\(Int(result.probability * 100))%")
                        .font(.subheadline.bold().monospaced())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            Capsule()
                                .fill(.white.opacity(0.22))
                        )
                }
            }
            .foregroundStyle(.white)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(tierColor.gradient)
            )
            .padding(.horizontal)

            // Explanation card
            VStack(alignment: .leading, spacing: 12) {
                Label("What this means", systemImage: "info.circle.fill")
                    .font(.subheadline.bold())
                Text(result == nil
                     ? "The analysis could not be completed. Please try again with a different image."
                     : "This is not a diagnosis. The tier reflects patterns the model found notable, not confirmed findings.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Divider()

                Label("Next step", systemImage: "stethoscope")
                    .font(.subheadline.bold())
                Text("Please consult a healthcare professional for any concerns.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(.secondarySystemBackground))
            )
            .padding(.horizontal)

            Spacer()

            Button {
                onDone()
            } label: {
                Label("Scan Another Image", systemImage: "arrow.counterclockwise")
                    .frame(maxWidth: .infinity)
                    .padding(6)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)
        }
        .padding(.bottom, 8)
    }
}

// MARK: - Image Picker (UIKit bridge)

struct ImagePicker: UIViewControllerRepresentable {
    let onImagePicked: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.onImagePicked(image)
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

#Preview {
    ContentView()
}
