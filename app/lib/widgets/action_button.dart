import 'package:flutter/material.dart';
import '../constants.dart';

/// 하단 액션 버튼 (upscaling, download)
class ActionButton extends StatelessWidget {
  final String text;
  final IconData icon;
  final VoidCallback onPressed;

  const ActionButton({
    super.key,
    required this.text,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: AppStyles.actionButtonStyle(),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon),
          const SizedBox(width: 8),
          Text(text, style: AppConstants.actionButtonTextStyle),
        ],
      ),
    );
  }
}

/// 하단 고정 버튼 래퍼
class BottomActionButton extends StatelessWidget {
  final Widget child;

  const BottomActionButton({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 20,
      right: 16,
      child: child,
    );
  }
}
