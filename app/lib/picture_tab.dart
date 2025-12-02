import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:contact/media_store_saver.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class PictureTab extends StatefulWidget {
  const PictureTab({super.key});

  @override
  State<PictureTab> createState() => _PictureTabState();
}

class _PictureTabState extends State<PictureTab> {
  Uint8List? _inputBytes;
  Uint8List? _outputBytes;
  String _inferenceTime = "";
  String? _inputResolution;
  String? _outputResolution;

  Future<void> _pickImage() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? picked = await picker.pickImage(source: ImageSource.gallery);

      if (picked != null) {
        final bytes = await picked.readAsBytes();
        setState(() {
          _inputBytes = bytes;
          _outputBytes = null;
          _inferenceTime = "";
          _inputResolution = null;
          _outputResolution = null;
        });

        ui.decodeImageFromList(bytes, (ui.Image image) {
          if (mounted) {
            setState(() {
              _inputResolution = "${image.width} x ${image.height}";
            });
          }
          image.dispose();
        });
      }
    } catch (e) {
      print("이미지 선택 에러: $e");
    }
  }

  Future<void> _convertImage() async {
    if (_inputBytes == null) return;

    final stopwatch = Stopwatch()..start();

    // 이미지 변환 로직
    await Future.delayed(const Duration(milliseconds: 300));
    _outputBytes = _inputBytes;

    if (_outputBytes != null) {
      ui.decodeImageFromList(_outputBytes!, (ui.Image image) {
        if (mounted) {
          setState(() {
            _outputResolution = "${image.width} x ${image.height}";
          });
        }
        image.dispose();
      });
    }

    stopwatch.stop();

    setState(() {
      _inferenceTime = "${stopwatch.elapsedMilliseconds} ms";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: SingleChildScrollView(
            child: Column(
              children: [
                const SizedBox(height: 15),
                // 입력 이미지
                Container(
                  height: 300,
                  width: double.infinity,
                  margin: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    shape: BoxShape.rectangle,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Stack(
                    children: [
                      GestureDetector(
                        onTap: _pickImage,
                        child: SizedBox.expand(
                          child: _inputBytes == null
                              ? Center(
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 24,
                                      vertical: 12,
                                    ),
                                    decoration: BoxDecoration(
                                      color: Colors.grey,
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: const Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(
                                          Icons.file_upload_outlined,
                                          color: Colors.white,
                                          size: 24,
                                        ),
                                        SizedBox(width: 8),
                                        Text(
                                          "image upload",
                                          style: TextStyle(
                                            color: Colors.white,
                                            fontWeight: FontWeight.w600,
                                            fontSize: 16,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                )
                              : ClipRRect(
                                  borderRadius: BorderRadius.circular(10),
                                  child: Image.memory(
                                    _inputBytes!,
                                    fit: BoxFit.contain,
                                  ),
                                ),
                        ),
                      ),
                      if (_inputResolution != null)
                        Positioned(
                          top: 12,
                          left: 12,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.3),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              _inputResolution!,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),

                // 출력 이미지
                Container(
                  height: 300,
                  width: double.infinity,
                  margin: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    shape: BoxShape.rectangle,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Stack(
                    children: [
                      SizedBox.expand(
                        child: _outputBytes == null
                            ? const SizedBox.shrink()
                            : ClipRRect(
                                borderRadius: BorderRadius.circular(10),
                                child: Image.memory(
                                  _outputBytes!,
                                  fit: BoxFit.contain,
                                ),
                              ),
                      ),
                      if (_outputResolution != null)
                        Positioned(
                          top: 12,
                          left: 12,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.3),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              _outputResolution!,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
                
                // 하단 여백 (버튼이 가리지 않도록)
                const SizedBox(height: 100),
              ],
            ),
          ),
        ),

        // 하단 고정 컨트롤 (Inference Time + Button)
        Positioned(
          bottom: 20,
          left: 0,
          right: 16, // 오른쪽 여백
          child: SizedBox(
            height: 60,
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Inference Time (가운데 정렬)
                // if (_inferenceTime.isNotEmpty) // 제거: 항상 보이도록 하거나 빈 문자열일 때는 공간만 차지하게 할 수 있음
                  Text(
                    "Inference time: $_inferenceTime",
                    style: const TextStyle(fontSize: 16),
                  ),
                
                // 버튼 (오른쪽 정렬)
                Align(
                  alignment: Alignment.centerRight,
                  child: _outputBytes == null
                      ? ElevatedButton(
                          onPressed: _convertImage,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 24,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.cached_rounded),
                              SizedBox(width: 8),
                              Text(
                                "image upscaling",
                                style: TextStyle(
                                    fontSize: 16, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                        )
                      : ElevatedButton(
                          onPressed: () async {
                            bool ok = await MediaStoreSaver.saveImage(_outputBytes!);
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text(ok ? "갤러리에 저장됨!" : "저장 실패")),
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 24,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.download_rounded),
                              SizedBox(width: 8),
                              Text(
                                "image download",
                                style: TextStyle(
                                    fontSize: 16, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                        ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
