; Inno Setup Script for FaceComparator
; UTF-8 encoding recommended

#define MyAppName "FaceComparator"
#define MyAppVersion "1.0"
#define MyAppPublisher "Mehmet Yüksel Şekeroğlu"
#define MyAppURL "https://github.com/MehmetYukselSekeroglu/FaceComparator-InsightFace-CXX"
#define MyAppExeName "FaceComparator.exe"
#define BuildPath "YourBuildDir"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{E3B0C442-98FC-1C14-AE4E-0D0A0A0A0A0A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Remove the following line to run in administrative install mode (install for all users.)
PrivilegesRequired=lowest
OutputDir={#BuildPath}\Installer
OutputBaseFilename=FaceComparator_Setup
SetupIconFile={#BuildPath}\app_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Ana uygulama dosyası
Source: "{#BuildPath}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Uygulama ikonu
Source: "{#BuildPath}\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

; Gerekli DLL'ler
Source: "{#BuildPath}\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\opencv_videoio_ffmpeg4120_64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\opencv_world4120.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\D3Dcompiler_47.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\opengl32sw.dll"; DestDir: "{app}"; Flags: ignoreversion

; Qt DLL'leri
Source: "{#BuildPath}\Qt6Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\Qt6Gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\Qt6Network.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\Qt6Pdf.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\Qt6Svg.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildPath}\Qt6Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion

; Qt eklenti klasörleri
Source: "{#BuildPath}\generic\*"; DestDir: "{app}\generic"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\iconengines\*"; DestDir: "{app}\iconengines"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\imageformats\*"; DestDir: "{app}\imageformats"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\networkinformation\*"; DestDir: "{app}\networkinformation"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\platforms\*"; DestDir: "{app}\platforms"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\styles\*"; DestDir: "{app}\styles"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\tls\*"; DestDir: "{app}\tls"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#BuildPath}\translations\*"; DestDir: "{app}\translations"; Flags: ignoreversion recursesubdirs createallsubdirs

; Models klasörü
Source: "{#BuildPath}\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.png"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.png"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.png"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Kurulum tamamlandığında models klasörünün varlığını kontrol et
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not DirExists(ExpandConstant('{app}\models')) then
    begin
      MsgBox('Models klasörü bulunamadı! Uygulama düzgün çalışmayabilir.', mbInformation, MB_OK);
    end;
  end;
end;