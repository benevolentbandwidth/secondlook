package com.hexogen.secondlook.ui.theme

import androidx.compose.ui.graphics.Color

// Apple's system colors, so this app renders the same as the SwiftUI build.
// Each name maps to the UIKit/SwiftUI color the iOS app asks for:
//
//   ContentView       iOS color                 constant below
//   ---------------   -----------------------   ----------------------
//   background top    .systemBackground         SystemBackground*
//   background bottom .systemGray6              SystemGray6*
//   cards             .secondarySystemBackground SecondarySystemBackground*
//   .tint / accent    accentColor (system blue) SystemBlue*
//   tier Low          .green                    SystemGreen*
//   tier Moderate     .orange                   SystemOrange*
//   tier Elevated     .red                      SystemRed*
//   unavailable       .gray                     SystemGray*
//   .foregroundStyle(.secondary)                SecondaryLabel*

// --- Light ---
val SystemBackgroundLight = Color(0xFFFFFFFF)
val SystemGray6Light = Color(0xFFF2F2F7)
val SecondarySystemBackgroundLight = Color(0xFFF2F2F7)
val LabelLight = Color(0xFF000000)
val SecondaryLabelLight = Color(0x993C3C43)   // #3C3C43 @ 60%
val SeparatorLight = Color(0x4A3C3C43)        // #3C3C43 @ 29%

val SystemBlueLight = Color(0xFF007AFF)
val SystemGreenLight = Color(0xFF34C759)
val SystemOrangeLight = Color(0xFFFF9500)
val SystemRedLight = Color(0xFFFF3B30)
val SystemGrayLight = Color(0xFF8E8E93)
val SwitchOffTrackLight = Color(0xFFE9E9EA)

// --- Dark ---
val SystemBackgroundDark = Color(0xFF000000)
val SystemGray6Dark = Color(0xFF1C1C1E)
val SecondarySystemBackgroundDark = Color(0xFF1C1C1E)
val LabelDark = Color(0xFFFFFFFF)
val SecondaryLabelDark = Color(0x99EBEBF5)    // #EBEBF5 @ 60%
val SeparatorDark = Color(0xA6545458)         // #545458 @ 65%

val SystemBlueDark = Color(0xFF0A84FF)
val SystemGreenDark = Color(0xFF30D158)
val SystemOrangeDark = Color(0xFFFF9F0A)
val SystemRedDark = Color(0xFFFF453A)
val SystemGrayDark = Color(0xFF8E8E93)
val SwitchOffTrackDark = Color(0xFF39393D)
