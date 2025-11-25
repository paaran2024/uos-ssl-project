package com.example.emutest

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

import android.content.ContentValues
import android.provider.MediaStore
import android.os.Build
import android.util.Log

class MainActivity: FlutterActivity() {

    private val CHANNEL = "media_store_saver"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "saveImage" -> {
                    val bytes = call.argument<ByteArray>("bytes")
                    if (bytes == null) {
                        result.error("NO_BYTES", "No image data received", null)
                        return@setMethodCallHandler
                    }
                    val ok = saveImage(bytes)
                    result.success(ok)
                }

                "saveVideo" -> {
                    val bytes = call.argument<ByteArray>("bytes")
                    if (bytes == null) {
                        result.error("NO_BYTES", "No video data received", null)
                        return@setMethodCallHandler
                    }
                    val ok = saveVideo(bytes)
                    result.success(ok)
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun saveImage(bytes: ByteArray): Boolean {
        return try {
            val resolver = contentResolver

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "IMG_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MyFlutterApp")
            }

            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                resolver.openOutputStream(it)?.use { out ->
                    out.write(bytes)
                }
            }
            true

        } catch (e: Exception) {
            Log.e("MediaStore", "Image save error", e)
            false
        }
    }

    private fun saveVideo(bytes: ByteArray): Boolean {
        return try {
            val resolver = contentResolver

            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, "VID_${System.currentTimeMillis()}.mp4")
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/MyFlutterApp")
            }

            val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                resolver.openOutputStream(it)?.use { out ->
                    out.write(bytes)
                }
            }
            true

        } catch (e: Exception) {
            Log.e("MediaStore", "Video save error", e)
            false
        }
    }
}
