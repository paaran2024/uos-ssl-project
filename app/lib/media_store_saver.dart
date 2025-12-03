import 'dart:typed_data';
import 'package:flutter/services.dart';

class MediaStoreSaver {
  static const MethodChannel _channel = MethodChannel("media_store_saver");

  static Future<bool> saveImage(Uint8List bytes) async {
    final ok = await _channel.invokeMethod("saveImage", {
      "bytes": bytes,
    });
    return ok == true;
  }

  static Future<bool> saveVideo(Uint8List bytes) async {
    final ok = await _channel.invokeMethod("saveVideo", {
      "bytes": bytes,
    });
    return ok == true;
  }
}
