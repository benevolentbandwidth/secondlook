package com.hexogen.secondlook

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.hexogen.secondlook.ui.SecondLookApp
import com.hexogen.secondlook.ui.theme.SecondLookTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SecondLookTheme {
                SecondLookApp()
            }
        }
    }
}
