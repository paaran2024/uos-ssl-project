import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';

class VideoTab extends StatefulWidget {
  const VideoTab({super.key});

  @override
  State<VideoTab> createState() => _VideoTabState();
}

class _VideoTabState extends State<VideoTab> {
  XFile? _inputVideo;
  XFile? _outputVideo;
  String _inferenceTime = "";

  VideoPlayerController? _inputController;
  VideoPlayerController? _outputController;

  // 비디오 선택
  Future<void> _pickVideo() async {
    final ImagePicker picker = ImagePicker();
    final XFile? picked = await picker.pickVideo(source: ImageSource.gallery);

    if (picked != null) {
      setState(() {
        _inputVideo = picked;
        _outputVideo = null;   // 출력 초기화
        _inferenceTime = "";
      });

      _inputController?.dispose();
      _inputController = VideoPlayerController.file(File(picked.path))
        ..initialize().then((_) {
          setState(() {});
        });
    }
  }

  // 변환
  Future<void> _convertVideo() async {
    if (_inputVideo == null) return;

    final stopwatch = Stopwatch()..start();

    // AI 모델 처리
    await Future.delayed(const Duration(milliseconds: 300));

    _outputVideo = _inputVideo;

    stopwatch.stop();
    _inferenceTime = "${stopwatch.elapsedMilliseconds} ms";

    _outputController?.dispose();
    _outputController =
    VideoPlayerController.file(File(_outputVideo!.path))
      ..initialize().then((_) {
        setState(() {});
      });

    setState(() {});
  }

  @override
  void dispose() {
    _inputController?.dispose();
    _outputController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(

      children: [
        const SizedBox(height: 15),

        //  입력 비디오
        GestureDetector(
          onTap: _pickVideo,
          child: Container(
            height: 250,
            width: double.infinity,
            margin: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(10),
            ),
            child: _inputVideo == null
                ? const Center(child: Text("입력 영상"))
                : _inputController != null &&
                _inputController!.value.isInitialized
                ? ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: AspectRatio(
                aspectRatio: _inputController!.value.aspectRatio,
                child: VideoPlayer(_inputController!),
              ),
            )
                : const Center(child: CircularProgressIndicator()),
          ),
        ),

        //  출력 비디오
        Container(
          height: 250,
          width: double.infinity,
          margin: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.grey.shade300,
            borderRadius: BorderRadius.circular(10),
          ),
          child: _outputVideo == null
              ? const Center(child: Text("출력 영상"))
              : _outputController != null &&
              _outputController!.value.isInitialized
              ? ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: AspectRatio(
              aspectRatio: _outputController!.value.aspectRatio,
              child: VideoPlayer(_outputController!),
            ),
          )
              : const Center(child: CircularProgressIndicator()),
        ),

        const SizedBox(height: 20),

        //  변환 버튼
        ElevatedButton(
          onPressed: _convertVideo,
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
