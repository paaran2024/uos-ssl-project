import 'dart:typed_data';
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

  Future<void> _pickImage() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? picked = await picker.pickImage(
        source: ImageSource.gallery,
      );

      if (picked != null) {
        final bytes = await picked.readAsBytes();
        setState(() {
          _inputBytes = bytes;
          _outputBytes = null;
          _inferenceTime = "";
        });
      }
    } catch (e) {
      print("이미지 선택 에러: $e");
    }
  }

  Future<void> _convertImage() async {
    if (_inputBytes == null) return;

    final stopwatch = Stopwatch()..start();

    await Future.delayed(const Duration(milliseconds: 300));
    _outputBytes = _inputBytes;

    stopwatch.stop();

    setState(() {
      _inferenceTime = "${stopwatch.elapsedMilliseconds} ms";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const SizedBox(height: 15),
        Row(
          children: [
            // 입력 이미지
            Expanded(
              child: GestureDetector(
                onTap: _pickImage,
                child: Container(
                  height: 300,
                  margin: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: _inputBytes == null
                      ? const Center(child: Text("입력 이미지"))
                      : ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image.memory(_inputBytes!, fit: BoxFit.cover),
                  ),
                ),
              ),
            ),

            // 출력 이미지
            Expanded(
              child: Container(
                height: 300,
                margin: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: _outputBytes == null
                    ? const Center(child: Text("출력 이미지"))
                    : ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.memory(_outputBytes!, fit: BoxFit.cover),
                ),
              ),
            ),
          ],
        ),

        const SizedBox(height: 20),

        //  버튼
        ElevatedButton(
          onPressed: _convertImage,
          child: const Text("변환"),
        ),

        const SizedBox(height: 16),

        //  Inference time
        Text(
          "inference time: $_inferenceTime",
          style: const TextStyle(fontSize: 16),
        ),

        const SizedBox(height: 20),
      ],
    );
  }
}
