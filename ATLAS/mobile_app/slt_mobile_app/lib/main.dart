import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'shell/app_shell.dart';
import 'shell/style.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.dark);
  runApp(const SignLanguageTranslationApp());
}

class SignLanguageTranslationApp extends StatelessWidget {
  const SignLanguageTranslationApp({super.key, this.splash = const Duration(milliseconds: 1400)});

  final Duration splash;

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'ATLAS',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      brightness: Brightness.light,
      fontFamily: Atlas.font,
      scaffoldBackgroundColor: Atlas.white,
      colorScheme: ColorScheme.fromSeed(
        seedColor: Atlas.blue,
        primary: Atlas.blue,
        surface: Atlas.white,
      ),
      useMaterial3: true,
    ),
    home: AppShell(splash: splash),
  );
}
