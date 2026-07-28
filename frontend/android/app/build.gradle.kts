plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.chaquopy)
}

android {
    namespace = "com.hexogen.secondlook"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.hexogen.secondlook"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Chaquopy packs a CPython runtime plus native NumPy/OpenCV per ABI.
        // Restrict to the two ABIs that matter (real devices + emulator) —
        // each extra ABI adds tens of megabytes to the APK.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // TFLite models are memory-mapped from the APK, so they must be stored
        // uncompressed.
        noCompress += "tflite"
    }
}

chaquopy {
    defaultConfig {
        // 3.10 is the only Python version with OpenCV wheels in Chaquopy's
        // package repository, and the preprocessing pipeline is cv2-based.
        // buildPython must be a local Python 3.10 — see README.
        version = "3.10"

        // Chaquopy looks for `python3.10` on PATH. Android Studio launched from
        // the Dock does not always inherit a shell PATH, so allow an override:
        //   ./gradlew -PchaquopyBuildPython=/opt/homebrew/bin/python3.10 ...
        // or set chaquopyBuildPython in gradle.properties / local.properties.
        providers.gradleProperty("chaquopyBuildPython").orNull?.let { buildPython(it) }

        pip {
            // Versions are the newest cp310 Android wheels Chaquopy publishes.
            install("numpy==1.26.2")
            install("opencv-python==4.5.1.48")
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.exifinterface)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.tensorflow.lite)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}

// Refresh the vendored copies of the Python pipeline from the Second Look
// research repo. The copies under src/main/python are the ones that ship; this
// task just keeps them from silently drifting. The default resolves to the repo
// root two levels up (frontend/android/ -> repo root); override with
// -PsecondLookRepo=... for a checkout somewhere else.
val secondLookRepo = providers.gradleProperty("secondLookRepo")
    .orElse(rootProject.layout.projectDirectory.dir("../..").asFile.path)

tasks.register<Copy>("syncPythonPipeline") {
    group = "second look"
    description = "Copy preprocessing sources from the Second Look Python repo into src/main/python."

    val repoDir = file(secondLookRepo.get())
    onlyIf {
        repoDir.isDirectory.also {
            if (!it) logger.warn("Second Look repo not found at $repoDir — nothing to sync.")
        }
    }

    from(repoDir) {
        include("config/constants.py")
        include("data_pipeline/_imaging_utils.py")
        include("data_pipeline/preprocessor.py")
        include("data_pipeline/label_mapper.py")
    }
    into(layout.projectDirectory.dir("src/main/python"))
}
