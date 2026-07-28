package com.hexogen.secondlook

import android.app.Application
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

/**
 * Starts the embedded CPython runtime once per process.
 *
 * Chaquopy extracts its stdlib and the NumPy/OpenCV wheels on first launch, so
 * doing this at application start keeps the cost off the scanning path.
 */
class SecondLookApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }
}
