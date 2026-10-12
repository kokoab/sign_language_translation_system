plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.slt_mobile_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.slt_mobile_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        // MediaPipe Tasks 0.10.14 ships ARM and x86 (no x86_64) native code; phones are ARM.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    // Live Reel models are memory-mapped from the APK: keep them uncompressed.
    androidResources {
        noCompress += listOf("tflite", "task")
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Live Reel (Android MediaPipe family): MediaPipe pinned to the desktop parity version; LiteRT 2.2 CompiledModel (CPU/XNNPACK, explicit threads).
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("androidx.activity:activity-ktx:1.10.1")
}
