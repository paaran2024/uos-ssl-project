import 'package:flutter/material.dart';
import '../constants.dart';

/// 미디어(이미지/비디오) 표시용 공통 컨테이너
class MediaContainer extends StatelessWidget {
  final Widget child;
  final String? resolution;
  final VoidCallback? onTap;

  const MediaContainer({
    super.key,
    required this.child,
    this.resolution,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: AppConstants.mediaContainerHeight,
        width: double.infinity,
        margin: AppConstants.containerMargin,
        decoration: AppStyles.mediaContainerDecoration(),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(AppConstants.containerBorderRadius),
          child: Stack(
            children: [
              SizedBox.expand(child: child),
              if (resolution != null) ResolutionBadge(resolution: resolution!),
            ],
          ),
        ),
      ),
    );
  }
}

/// 해상도 표시 배지
class ResolutionBadge extends StatelessWidget {
  final String resolution;

  const ResolutionBadge({super.key, required this.resolution});

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 12,
      left: 12,
      child: Container(
        padding: AppConstants.resolutionBadgePadding,
        decoration: AppStyles.resolutionBadgeDecoration(),
        child: Text(
          resolution,
          style: AppConstants.resolutionTextStyle,
        ),
      ),
    );
  }
}

/// 업로드 플레이스홀더 버튼
class UploadPlaceholder extends StatelessWidget {
  final String text;
  final IconData icon;

  const UploadPlaceholder({
    super.key,
    required this.text,
    this.icon = Icons.file_upload_outlined,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        padding: AppConstants.uploadButtonPadding,
        decoration: AppStyles.uploadButtonDecoration(),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 24),
            const SizedBox(width: 8),
            Text(text, style: AppConstants.uploadButtonTextStyle),
          ],
        ),
      ),
    );
  }
}
