import 'package:flutter/services.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';

class MediaStoreSaver {
  static Future<bool> saveImage(Uint8List bytes) async {
    final result = await ImageGallerySaver.saveImage(bytes, quality: 100);
    return result['isSuccess'] == true;
  }

  static Future<bool> saveVideo(Uint8List bytes) async {
    // 비디오는 파일로 먼저 저장 후 갤러리에 추가해야 함
    return false;
  }
}
