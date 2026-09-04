# Signa Android App

This folder wraps the existing `index.html` recognition UI in a native Android WebView. The TensorFlow.js model and labels are bundled under `app/src/main/assets/models/`.

## Build the APK

1. Install Android Studio with the Android SDK and SDK Platform 35.
2. Open this `android_app` folder in Android Studio.
3. Let Gradle sync and install any suggested Android SDK components.
4. Select **Build > Build APK(s)**.
5. The debug APK will be at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Install it on a connected phone with:

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

The app requests camera permission on first launch. It also needs internet access because the current HTML loads TensorFlow.js and MediaPipe from a CDN. The recognition model itself is bundled locally.
