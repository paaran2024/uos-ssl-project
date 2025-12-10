import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';

import 'constants.dart';
import 'widgets/widgets.dart';
import 'services/services.dart';
import 'media_store_saver.dart';

class VideoTab extends StatefulWidget {
  const VideoTab({super.key});

  @override
  State<VideoTab> createState() => _VideoTabState();
}

class _VideoTabState extends State<VideoTab> {
  XFile? _inputVideo;
  XFile? _outputVideo;
  String _inferenceTime = "";
  String? _inputResolution;
  String? _outputResolution;

  VideoPlayerController? _inputController;
  VideoPlayerController? _outputController;

  @override
  void dispose() {
    _inputController?.dispose();
    _outputController?.dispose();
    super.dispose();
  }

  Future<void> _pickVideo() async {
    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: ImageSource.gallery);

    if (picked == null) return;

    setState(() {
      _inputVideo = picked;
      _outputVideo = null;
      _inferenceTime = "";
      _inputResolution = null;
      _outputResolution = null;
    });

    _inputController?.dispose();
    _inputController = VideoPlayerController.file(File(picked.path))
      ..initialize().then((_) {
        if (mounted) {
          setState(() {
            final size = _inputController!.value.size;
            _inputResolution = "${size.width.toInt()} x ${size.height.toInt()}";
          });
        }
      });
  }

  Future<void> _convertVideo() async {
    if (_inputVideo == null) return;

    final result = await VideoConverter.convert(_inputVideo!);

    setState(() {
      _outputVideo = result.outputVideo;
      _inferenceTime = "${result.inferenceTimeMs} ms";
    });

    _outputController?.dispose();
    _outputController = VideoPlayerController.file(File(_outputVideo!.path))
      ..initialize().then((_) {
        if (mounted) {
          setState(() {
            final size = _outputController!.value.size;
            _outputResolution = "${size.width.toInt()} x ${size.height.toInt()}";
          });
        }
      });
  }

  Future<void> _saveVideo() async {
    if (_outputVideo == null) return;

    final Uint8List bytes = await _outputVideo!.readAsBytes();
    final ok = await MediaStoreSaver.saveVideo(bytes);

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(ok ? "갤러리에 저장됨" : "저장 실패")),
    );
  }

  void _openFullScreen(VideoPlayerController controller) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => FullScreenVideoPlayer(controller: controller),
      ),
    );
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
                _buildInputVideoContainer(),
                _buildOutputVideoContainer(),
                const SizedBox(height: 30),
                _buildInferenceTime(),
                const SizedBox(height: 100),
              ],
            ),
          ),
        ),
        BottomActionButton(child: _buildActionButton()),
      ],
    );
  }

  Widget _buildInputVideoContainer() {
    return _buildVideoContainer(
      controller: _inputController,
      resolution: _inputResolution,
      placeholderText: "video upload",
      isInput: true,
      onTap: _pickVideo,
    );
  }

  Widget _buildOutputVideoContainer() {
    return _buildVideoContainer(
      controller: _outputController,
      resolution: _outputResolution,
      placeholderText: "output video",
      isInput: false,
      onTap: null,
    );
  }

  Widget _buildVideoContainer({
    required VideoPlayerController? controller,
    required String? resolution,
    required String placeholderText,
    required bool isInput,
    required VoidCallback? onTap,
  }) {
    // 컨트롤러 없음: 플레이스홀더 또는 빈 컨테이너
    if (controller == null) {
      return MediaContainer(
        resolution: resolution,
        onTap: isInput ? onTap : null,
        child: isInput
            ? UploadPlaceholder(text: placeholderText)
            : const SizedBox.shrink(),
      );
    }

    // 초기화 중: 로딩 표시
    if (!controller.value.isInitialized) {
      return MediaContainer(
        child: const Center(child: CircularProgressIndicator()),
      );
    }

    // 초기화 완료: 비디오 플레이어 표시
    return MediaContainer(
      resolution: resolution,
      onTap: onTap,
      child: _buildVideoPlayer(controller, isInput),
    );
  }

  Widget _buildVideoPlayer(VideoPlayerController controller, bool isInput) {
    return Stack(
      alignment: Alignment.center,
      children: [
        AspectRatio(
          aspectRatio: controller.value.aspectRatio,
          child: VideoPlayer(controller),
        ),
        _buildPlayPauseButton(controller),
        if (!isInput) _buildFullScreenButton(controller),
      ],
    );
  }

  Widget _buildPlayPauseButton(VideoPlayerController controller) {
    return IconButton(
      icon: Icon(
        controller.value.isPlaying ? Icons.pause_circle : Icons.play_circle,
        size: 40,
        color: Colors.white,
      ),
      onPressed: () {
        setState(() {
          controller.value.isPlaying ? controller.pause() : controller.play();
        });
      },
    );
  }

  Widget _buildFullScreenButton(VideoPlayerController controller) {
    return Positioned(
      bottom: 10,
      right: 10,
      child: IconButton(
        icon: const Icon(Icons.fullscreen, size: 36, color: Colors.white),
        onPressed: () => _openFullScreen(controller),
      ),
    );
  }

  Widget _buildInferenceTime() {
    return Text(
      "Inference time: $_inferenceTime",
      style: AppConstants.inferenceTimeTextStyle,
    );
  }

  Widget _buildActionButton() {
    if (_outputVideo == null) {
      return ActionButton(
        text: "video upscaling",
        icon: Icons.cached_rounded,
        onPressed: _convertVideo,
      );
    }

    return ActionButton(
      text: "video download",
      icon: Icons.download_rounded,
      onPressed: _saveVideo,
    );
  }
}
