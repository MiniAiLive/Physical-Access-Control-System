
<div align="center">
   <h1>Physical Access Control System</h1>
   <img src=https://miniai.live/wp-content/uploads/2024/02/logo_name-1-768x426-1.png alt="MiniAiLive Logo"
   width="300">
</div>

## Welcome to the [MiniAiLive](https://www.miniai.live/)!
This is the world first repository which describes full solutions for Physical Access Control System containing from hardware design, Face Recognition & Face Liveness Detection (3D Face Passive Anti-spoofing) model to deployment for device.
Can be applied to face recognition based smart-lock or similar solution easily.

## Introduction

This repository contains the source code and documentation for a physical access control system. The access control device is a facial recognition access control device that uses two-camera 3D facial recognition technology to control access. The system is designed to manage and control access to physical spaces using RV1108, F1C200s, and Nuvoton chip. Also, provides physical enclosure for physical system so that you can build full physical access control system by your own.

<!-- ## System Configuration
<img src="https://github.com/MiniAiLive/MiniAI-Physical-Access-Control-System/blob/main/Assets/Picture1.png" alt="Test1" width="650" />

## Real System

<img src="https://github.com/MiniAiLive/MiniAI-Physical-Access-Control-System/blob/main/Assets/RealImage.png" alt="Test1" width="750" /> -->

## Dominance
- A single stereo vision 3D camera can be used for face recognition and video chat.
- Stereo vision 3D face recognition function
- Video and voice conversation functions
- Uninterruptible power supply function
- 1:1 encrypted communication between outdoor unit and power control box
- Human body confinement function
- RS485/RS232, Wiegand support

## Technical characteristics

- CPU 1.0GHz, high-performance DSP support
- Operating system LINUX
- Basic functions: door opening function, video conversation function, record management function
- Fake identification and prevention of various types of fake attacks such as 2D photos, videos, 3D models, masks, etc.
- Usage environment Recognition possible in various lighting environments such as indoors, darkroom, and outdoors
- Camera module stereo vision 3D camera (2 infrared cameras + 1 color camera)
- Face recognition performance FAR≤0.00001%, FRR≤0.01%
- Stereovision 3D face recognition using deep learning algorithm, automatic adaptation to facial changes
- Recognition distance 30~100cm
- Authentication time less than 0.5 seconds
- Door opening method: face, card, password (two-factor authentication function supported)
- Opening the door using the door opening button of the indoor unit
- Number of registered people: 1000
- Enclosure size 129 * 107 * 40mm
- LCD outdoor unit LCD: 3.5 inch LCD
- Indoor unit LCD: 3.5 inch LCD
- 3.5 inch touch panel
- Human body confinement distance of 1m or more
- Power supply characteristics Inlet voltage: 12V
- 24-hour power supply method
- Permission management administrator, user
- Operating temperature -25 degrees to 70 degrees
- Video chat function enables doorstep surveillance and simultaneous two-way voice calls within the door

## Basic functions

It supports video and voice conversation functions along with door opening and record management functions.
A single stereo vision 3D camera module performs face recognition and video conversation functions.

1. **Door opening function:**
- There are registration and authentication functions using face, card, and password.
- Users can open the key by registering and authenticating using any of the following methods: face, card, or password.
- You can also open the key by pressing the door open button on the indoor unit.
2. **Record management function:**
- All authentication records are stored in the outdoor unit, and the stored records can be viewed in the outdoor unit's system guide - records management item.
- By selecting the USB-record output item in the outdoor unit settings, you can store all authentication records stored in the outdoor unit in XML file format along with photo data on a USB.
3. **Outdoor unit setting face-to-face function:**
- Outdoor unit setting functions include user management function, record viewing function, system setting function, date and time setting function, USB data input/output function, and system information viewing function.
- You can register, change, or delete user information.
- You can register as a user using any of the following methods: face, card, or password.
- User rights include administrator and user, and up to 1,000 people can be registered.
3. **Record Viewing function:**
- You can view all certification records. For facial authentication records, face photos are also displayed.
Up to 10,000 items can be stored.
4. **Language Setting:**
- Set all system menu and audio language.
- Setting value is english and chinese - default English right now.
- Can add any language.
- You can set the volume between 0-100% in 20% increments. The default value is 100%.
- You can set up two-factor authentication method/single authentication method. The default value is single authentication method.
5. **Camera display method setting function:**
- Set the camera display method during face registration and authentication.
- The settings are auto, color, and infrared. The default value is automatic.
6. **Screen Saver Function:**
- Set the time until the authentication screen switches to screen protection mode.
- You can set it in 1 minute intervals from 1 to 10 minutes. The default value is 3 minutes.
7. **Factory Setting Function:**
Return system settings to their initial state.

## Indoor Unit Setting
Set the display language for the indoor unit settings interface.
The settings are Chinese (Simplified), Chinese (Traditional), and English. The default is English.

**Video chat function:**
- The video conversation function is a function that allows you to monitor the front door from the indoor unit and conduct voice conversations from inside and outside the door. A video conversation can be started by pressing the power button on the indoor unit, or a guest can start it by pressing the doorbell on the outdoor unit.
Once you start a video conversation, you can proceed with video and voice conversations for the set time (you can set the timeout time in the indoor unit settings face). After the set time, the indoor unit liquid crystal display turns off and the outdoor unit liquid crystal display turns on, completing the video conversation.

## Functions of Buttons
1. **Power button:**
- If you want to monitor the outside of the door from inside the door, you can start a video conversation by pressing the power button.
- When you press the power button, the outdoor unit LCD turns off and the indoor unit LCD turns on, allowing you to monitor the outside of the door.
- If you press and hold the power button during a video conversation, the indoor unit LCD turns off and the outdoor unit LCD turns on, completing the video conversation.
- Additionally, each time you press the power button during a video conversation, the video conversation timeout coefficient is reset.
2. **Open button:**
- If you press the open button at any time, key opening will proceed.
- Additionally, the video chat timeout coefficient is reset each time the open button is pressed during a video chat.
- If you press and hold the Open button while the indoor unit settings interface is displayed, you will withdraw from the setup interface and enter video chat mode.
3. **Microphone button**
- When the video conversation starts, the microphone icon on the indoor unit liquid crystal display is set to inactive by default. In other words, when a video conversation starts, only one-way voice calls (outdoor unit - indoor unit) are enabled by default.
- When you press the microphone button during a video conversation, the microphone icon on the indoor unit's liquid crystal display becomes active, allowing simultaneous two-way voice calls between the indoor unit and the outdoor unit. If you press the microphone button again and the microphone icon becomes inactive, only one-way voice calls (outdoor unit - indoor unit) will be possible.
- If you press and hold the microphone button during a video conversation, the indoor unit settings screen will appear.
- Additionally, the video conversation timeout coefficient is reset each time the microphone button is pressed during a video conversation.

## GitHub Folder Structure
1. **3D Model:**
- Front folder contains 3D design file of front part of enclosure for this system. Support solidworks file and .step file.
- Back folder contains 3D design file of back part of enclosure for this system. Support solidworks file and .step file.
2. **AIStuff:**
- Contains all necessary part for face recognition and face liveness detection code for this system.
3. **Circuit:**
- Face_3Key1v0: Design of keyboard board which control 3 buttons of the indoor system.
- Face_Ant1v0: Design of Antenna board for NFC card recognition.
- Face_HM1v0: Design of human sensor board for waking device up automatically when user access to system.
- Face_InBack1v0: Design of indoor back board which use N76E003 chip.
- Face_InMain1v0: Design of main board which use F1C chip.
- Face_MainBack1v0: Design of outdoor back board.
- Face_PWR1v0: Design of power management board which use N76E003 chip.
4. **Final Firmwares:**
- 201102_F1C100s_VD5 1.0.6: F1C firmware file.
- N76E003_HM-1.02_20201014: Human sensor firmware file.
- N76E003_InMain_PM_1.03: Indoor Main Board firmeware file.
- N76E003_PowerControl_1.02-P_20200930: Power Control firmware file.
5. **Final Firmwares:**
- F1C200s: F1C firmware source code.
- N76E003/N76E003_InMainPMU: Indoor main board N76E003 source code.
- RV1108: Outdoor Main Board Source Code.

## Important Note
- This solution is already mass producted.
- The source code is missing some important part, we didn't push those code because it is not open-source.
- This solution can be applicable to similar project like face recognition based smart-lock, we are using this source code for our smart-lock mainly, please visit [here](https://www.youtube.com/watch?v=PxlNMM5rXlI) for confirmation.
<br>

## Face & IDSDK Online Demo, Resources
<div style="display: flex; justify-content: center; align-items: center;"> 
   <table style="text-align: center;">
      <tr>
         <td style="text-align: center; vertical-align: middle;"><a href="https://github.com/MiniAiLive"><img src="https://miniai.live/wp-content/uploads/2024/10/new_git-1-300x67.png" style="height: 60px; margin-right: 5px;" title="GITHUB"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://huggingface.co/MiniAiLive"><img src="https://miniai.live/wp-content/uploads/2024/10/new_hugging-1-300x67.png" style="height: 60px; margin-right: 5px;" title="HuggingFace"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://demo.miniai.live"><img src="https://miniai.live/wp-content/uploads/2024/10/new_gradio-300x67.png" style="height: 60px; margin-right: 5px;" title="Gradio"/></a></td> 
      </tr> 
      <tr>
         <td style="text-align: center; vertical-align: middle;"><a href="https://docs.miniai.live/"><img src="https://miniai.live/wp-content/uploads/2024/10/a-300x70.png" style="height: 60px; margin-right: 5px;" title="Documentation"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://www.youtube.com/@miniailive"><img src="https://miniai.live/wp-content/uploads/2024/10/Untitled-1-300x70.png" style="height: 60px; margin-right: 5px;" title="Youtube"/></a></td> 
         <td style="text-align: center; vertical-align: middle;"><a href="https://play.google.com/store/apps/dev?id=5831076207730531667"><img src="https://miniai.live/wp-content/uploads/2024/10/googleplay-300x62.png" style="height: 60px; margin-right: 5px;" title="Google Play"/></a></td>
      </tr>
   </table>
</div>

## Our Products
No | Project | Feature
---|---|---|
1 | [FaceRecognition-LivenessDetection-Android-SDK](https://github.com/MiniAiLive/FaceRecognition-LivenessDetection) | Face Matching, 3D Face Passive Liveness
2 | [FaceRecognition-LivenessDetection-Windows-SDK](https://github.com/MiniAiLive/FaceRecognition-LivenessDetection-Windows-SDK) | Face Matching, 3D Face Passive Liveness
3 | [FaceLivenessDetection-Android-SDK](https://github.com/MiniAiLive/FaceLivenessDetection-Android-SDK) | 3D Face Passive Liveness
4 | [FaceLivenessDetection-Linux-SDK](https://github.com/MiniAiLive/FaceLivenessDetection-Linux-SDK) | 3D Face Passive Liveness
5 | [FaceLivenessDetection-Windows-SDK](https://github.com/MiniAiLive/FaceLivenessDetection-Windows-SDK) | 3D Face Passive Liveness
6 | [FaceMatching-Android-SDK](https://github.com/MiniAiLive/FaceMatching-Android-SDK) | 1:1 Face Matching
7 | [FaceMatching-Windows-Demo](https://github.com/MiniAiLive/FaceMatching-Windows-Demo) | 1:1 Face Matching
8 | [FaceAttributes-Android-SDK](https://github.com/MiniAiLive/FaceAttributes-Android-SDK) | Face Attributes
9 | [ID-Document-Recognition-Android-SDK](https://github.com/MiniAiLive/ID-Document-Recognition-Android-SDK) | ID Document, Credit, MRZ Recognition
10 | [ID-Document-Recognition-Linux-SDK](https://github.com/MiniAiLive/ID-Document-Recognition-Linux-SDK) | ID Document, Credit, MRZ Recognition
11 | [ID-Document-Recognition-Windows-SDK](https://github.com/MiniAiLive/ID-Document-Recognition-Windows-SDK) | ID Document, Credit, MRZ Recognition
12 | [ID-Document-LivenessDetection-Linux-SDK](https://github.com/MiniAiLive/ID-Document-LivenessDetection-Linux-SDK) | ID Document Liveness Detection
13 | [ID-Document-LivenessDetection-Windows-SDK](https://github.com/MiniAiLive/ID-Document-LivenessDetection-Windows-SDK) | ID Document Liveness Detection

## Contact US
For requesting missing code and any customizable solution using this kind of techniques, please [Contact US](https://www.miniai.live/contact/)

<p align="center">
<a target="_blank" href="https://t.me/Contact_MiniAiLive"><img src="https://img.shields.io/badge/telegram-@MiniAiLive-blue.svg?logo=telegram" alt="www.miniai.live"></a>&emsp;
<a target="_blank" href="https://wa.me/+19162702374"><img src="https://img.shields.io/badge/whatsapp-MiniAiLive-blue.svg?logo=whatsapp" alt="www.miniai.live"></a>&emsp;
<a target="_blank" href="https://join.skype.com/invite/ltQEVDmVddTe"><img src="https://img.shields.io/badge/skype-MiniAiLive-blue.svg?logo=skype" alt="www.miniai.live"></a>&emsp;
</p>
