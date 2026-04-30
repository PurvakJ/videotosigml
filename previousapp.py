from flask import Flask, render_template, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import math
import json
from pathlib import Path
import warnings
import time
import os
import base64
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# =========================
# ENUMS AND CONSTANTS
# =========================

class HandShape(Enum):
    FIST = "fist"
    OPEN = "open"
    POINT = "point"
    VICTORY = "victory"
    L_SHAPE = "l_shape"
    OTHER = "other"
    UNKNOWN = "unknown"


class HandOrientation(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class PalmOrientation(Enum):
    PALM_UP = "PALM_UP"
    PALM_DOWN = "PALM_DOWN"
    PALM_LEFT = "PALM_LEFT"
    PALM_RIGHT = "PALM_RIGHT"
    PALM_FORWARD = "PALM_FORWARD"
    UNKNOWN = "UNKNOWN"


class MovementDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STILL = "STILL"
    UNKNOWN = "UNKNOWN"


class Location(Enum):
    HEAD = "head"
    UNDERCHIN = "underchin"
    HANDBACK = "handback"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class GestureType(Enum):
    TWO_HANDED_GESTURE = "TWO_HANDED_GESTURE"
    GESTURE_WITH_HEAD_MOVEMENT = "GESTURE_WITH_HEAD_MOVEMENT"
    HANDSHAPE_GESTURE = "HANDSHAPE_GESTURE"
    MANUAL_GESTURE = "MANUAL_GESTURE"
    LOCATION_GESTURE = "LOCATION_GESTURE"


# =========================
# DATA CLASSES
# =========================

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class FrameData:
    frame: int
    timestamp: float
    left_hand: Optional[List[Landmark]] = None
    right_hand: Optional[List[Landmark]] = None
    head: Optional[List[Landmark]] = None
    left_handshape: HandShape = HandShape.UNKNOWN
    right_handshape: HandShape = HandShape.UNKNOWN
    left_orientation: HandOrientation = HandOrientation.UNKNOWN
    right_orientation: HandOrientation = HandOrientation.UNKNOWN
    left_palm_orientation: PalmOrientation = PalmOrientation.UNKNOWN
    right_palm_orientation: PalmOrientation = PalmOrientation.UNKNOWN
    head_orientation: str = "NEUTRAL"
    left_movement: Optional[MovementDirection] = None
    right_movement: Optional[MovementDirection] = None
    head_movement: Optional[MovementDirection] = None
    left_wrist_pos: Optional[Landmark] = None
    right_wrist_pos: Optional[Landmark] = None
    nose_pos: Optional[Landmark] = None
    left_location: Location = Location.UNKNOWN
    right_location: Location = Location.UNKNOWN
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0


@dataclass
class Gesture:
    type: GestureType
    frame: int
    time: float
    left_shape: Optional[HandShape] = None
    right_shape: Optional[HandShape] = None
    hand_shape: Optional[HandShape] = None
    hand_orientation: Optional[HandOrientation] = None
    palm_orientation: Optional[PalmOrientation] = None
    left_orientation: Optional[HandOrientation] = None
    right_orientation: Optional[HandOrientation] = None
    left_palm_orientation: Optional[PalmOrientation] = None
    right_palm_orientation: Optional[PalmOrientation] = None
    movement: Optional[MovementDirection] = None
    hand_movement: Optional[MovementDirection] = None
    head_movement: Optional[MovementDirection] = None
    location: Optional[Location] = None
    hand: str = "left"
    confidence: float = 1.0
    description: str = ""


# =========================
# SIGN LANGUAGE ANALYZER CLASS
# =========================

class SignLanguageAnalyzer:
    """Main class for sign language analysis from video"""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize the analyzer with MediaPipe Holistic model"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Analysis data
        self.analysis_data = {
            'frames': [],
            'gestures': [],
            'sigml_signs': []
        }

        # Gesture buffer for temporal smoothing
        self.gesture_buffer = []
        self.gesture_buffer_size = 5
        self.previous_frame_data = None

        # Movement detection parameters
        self.MOVEMENT_THRESHOLD = 0.005
        self.HEAD_MOVEMENT_THRESHOLD = 0.003

        # Handshape stability tracking
        self.left_handshape_history = []
        self.right_handshape_history = []
        self.location_history = []
        self.history_size = 5

    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                      frame_skip: int = 1, callback=None):
        """
        Process a video file and detect sign language gestures
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        self.analysis_data = {'frames': [], 'gestures': [], 'sigml_signs': []}
        self.gesture_buffer = []
        self.previous_frame_data = None
        self.left_handshape_history = []
        self.right_handshape_history = []
        self.location_history = []

        frame_count = 0
        processed_count = 0

        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(frame_rgb)
                timestamp = frame_count / fps
                frame_data = self._process_frame_data(results, processed_count, timestamp)

                self.analysis_data['frames'].append(frame_data)
                self._detect_gestures_with_buffer(frame_data)

                processed_count += 1

                if callback and processed_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    callback(progress, frame_data)

            frame_count += 1

        cap.release()
        return self.analysis_data

    def _process_frame_data(self, results, frame_id: int, timestamp: float) -> FrameData:
        """Extract all relevant data from MediaPipe results"""
        data = FrameData(frame=frame_id, timestamp=timestamp)

        # Process left hand
        if results.left_hand_landmarks:
            data.left_hand = self._extract_landmarks(results.left_hand_landmarks.landmark)
            data.left_handshape = self._get_handshape(data.left_hand)
            data.left_orientation = self._get_hand_orientation(data.left_hand)
            data.left_palm_orientation = self._get_palm_orientation(data.left_hand)
            data.left_wrist_pos = data.left_hand[0]

        # Process right hand
        if results.right_hand_landmarks:
            data.right_hand = self._extract_landmarks(results.right_hand_landmarks.landmark)
            data.right_handshape = self._get_handshape(data.right_hand)
            data.right_orientation = self._get_hand_orientation(data.right_hand)
            data.right_palm_orientation = self._get_palm_orientation(data.right_hand)
            data.right_wrist_pos = data.right_hand[0]

        # Process pose
        if results.pose_landmarks:
            data.head = self._extract_landmarks(results.pose_landmarks.landmark)
            data.head_orientation = self._get_head_orientation(data.head)
            data.nose_pos = data.head[0]

        # Calculate movements compared to previous frame
        if self.previous_frame_data:
            if data.left_wrist_pos and self.previous_frame_data.left_wrist_pos:
                data.left_movement = self._calculate_movement_direction(
                    self.previous_frame_data.left_wrist_pos,
                    data.left_wrist_pos,
                    self.MOVEMENT_THRESHOLD
                )

            if data.right_wrist_pos and self.previous_frame_data.right_wrist_pos:
                data.right_movement = self._calculate_movement_direction(
                    self.previous_frame_data.right_wrist_pos,
                    data.right_wrist_pos,
                    self.MOVEMENT_THRESHOLD
                )

            if data.nose_pos and self.previous_frame_data.nose_pos:
                data.head_movement = self._calculate_movement_direction(
                    self.previous_frame_data.nose_pos,
                    data.nose_pos,
                    self.HEAD_MOVEMENT_THRESHOLD
                )

        self.previous_frame_data = data
        return data

    def _extract_landmarks(self, landmarks) -> List[Landmark]:
        """Convert MediaPipe landmarks to our Landmark class"""
        return [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, 'visibility', 1.0)
            )
            for lm in landmarks
        ]

    def _get_hand_orientation(self, landmarks: List[Landmark]) -> HandOrientation:
        """Determine hand orientation from wrist to middle finger tip"""
        if not landmarks or len(landmarks) < 21:
            return HandOrientation.UNKNOWN

        wrist = landmarks[0]
        middle_tip = landmarks[12]

        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y

        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return HandOrientation.UNKNOWN

        norm_dx = dx / length
        norm_dy = dy / length

        threshold = 0.3

        if abs(norm_dy) > abs(norm_dx):
            if norm_dy < -threshold:
                return HandOrientation.UP
            if norm_dy > threshold:
                return HandOrientation.DOWN
        else:
            if norm_dx < -threshold:
                return HandOrientation.LEFT
            if norm_dx > threshold:
                return HandOrientation.RIGHT

        return HandOrientation.NEUTRAL

    def _get_palm_orientation(self, landmarks: List[Landmark]) -> PalmOrientation:
        """Determine palm orientation from index and pinky MCP joints"""
        if not landmarks or len(landmarks) < 21:
            return PalmOrientation.UNKNOWN

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        dx = pinky_mcp.x - index_mcp.x
        dy = pinky_mcp.y - index_mcp.y

        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return PalmOrientation.UNKNOWN

        norm_dx = dx / length
        norm_dy = dy / length

        threshold = 0.3

        if abs(norm_dy) > abs(norm_dx):
            if norm_dy < -threshold:
                return PalmOrientation.PALM_UP
            if norm_dy > threshold:
                return PalmOrientation.PALM_DOWN
        else:
            if norm_dx < -threshold:
                return PalmOrientation.PALM_RIGHT
            if norm_dx > threshold:
                return PalmOrientation.PALM_LEFT

        return PalmOrientation.PALM_FORWARD

    def _get_head_orientation(self, landmarks: List[Landmark]) -> str:
        """Determine head orientation from pose landmarks"""
        if not landmarks or len(landmarks) < 33:
            return "NEUTRAL"

        nose = landmarks[0]
        left_eye = landmarks[2]
        right_eye = landmarks[5]

        eye_center_x = (left_eye.x + right_eye.x) / 2
        dx = nose.x - eye_center_x

        if abs(dx) > 0.02:
            return "RIGHT" if dx > 0 else "LEFT"
        return "NEUTRAL"

    def _calculate_movement_direction(self, prev_pos: Landmark,
                                      curr_pos: Landmark,
                                      threshold: float) -> MovementDirection:
        """Calculate movement direction between two points"""
        dx = curr_pos.x - prev_pos.x
        dy = curr_pos.y - prev_pos.y
        speed = math.sqrt(dx * dx + dy * dy)

        if speed < threshold:
            return MovementDirection.STILL

        if abs(dx) > abs(dy):
            return MovementDirection.RIGHT if dx > 0 else MovementDirection.LEFT
        return MovementDirection.DOWN if dy > 0 else MovementDirection.UP

    def _get_handshape(self, landmarks: List[Landmark]) -> HandShape:
        """Determine handshape from hand landmarks"""
        if not landmarks or len(landmarks) < 21:
            return HandShape.UNKNOWN

        # Calculate finger extended states
        finger_states = {
            'thumb': self._is_finger_extended(landmarks, [0, 2, 3, 4]),
            'index': self._is_finger_extended(landmarks, [0, 5, 6, 8]),
            'middle': self._is_finger_extended(landmarks, [0, 9, 10, 12]),
            'ring': self._is_finger_extended(landmarks, [0, 13, 14, 16]),
            'pinky': self._is_finger_extended(landmarks, [0, 17, 18, 20])
        }

        # Count extended fingers
        extended_count = sum(finger_states.values())

        # Determine handshape
        if extended_count == 0:
            return HandShape.FIST
        elif finger_states['index'] and not any([finger_states['middle'],
                                                 finger_states['ring'],
                                                 finger_states['pinky']]):
            return HandShape.POINT
        elif finger_states['index'] and finger_states['middle'] and not any([
            finger_states['ring'], finger_states['pinky']]):
            # Check if fingers are separated (V sign)
            tip_index = landmarks[8]
            tip_middle = landmarks[12]
            distance = abs(tip_index.x - tip_middle.x)
            if distance > 0.03:
                return HandShape.VICTORY
            else:
                return HandShape.OTHER
        elif finger_states['thumb'] and finger_states['index'] and not any([
            finger_states['middle'], finger_states['ring'], finger_states['pinky']]):
            return HandShape.L_SHAPE
        elif all(finger_states.values()):
            return HandShape.OPEN

        return HandShape.OTHER

    def _is_finger_extended(self, landmarks: List[Landmark], indices: List[int]) -> bool:
        """Check if a finger is extended based on joint angles"""
        try:
            wrist, mcp, pip, tip = [landmarks[i] for i in indices]

            angle1 = self._calculate_angle(wrist, mcp, pip)
            angle2 = self._calculate_angle(mcp, pip, tip)

            return angle1 > 150 and angle2 > 150
        except (IndexError, AttributeError):
            return False

    def _calculate_angle(self, a: Landmark, b: Landmark, c: Landmark) -> float:
        """Calculate angle between three points"""
        ab = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

        cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle) * 180 / np.pi

    def _detect_gestures_with_buffer(self, frame_data: FrameData):
        """Detect gestures using temporal buffer"""
        # Add to buffer
        self.gesture_buffer.append(frame_data)
        if len(self.gesture_buffer) > self.gesture_buffer_size:
            self.gesture_buffer.pop(0)

        # Analyze buffer for stable gestures
        if len(self.gesture_buffer) == self.gesture_buffer_size:
            self._analyze_gesture_buffer()

    def _analyze_gesture_buffer(self):
        """Analyze buffer to detect stable gestures"""
        if not self.gesture_buffer:
            return

        last_frame = self.gesture_buffer[-1]

        # Check for consistent movements across buffer
        consistent_left = True
        consistent_right = True
        consistent_head = True

        left_movement_dir = None
        right_movement_dir = None
        head_movement_dir = None

        for i in range(1, len(self.gesture_buffer)):
            prev = self.gesture_buffer[i - 1]
            curr = self.gesture_buffer[i]

            # Check left hand consistency
            if prev.left_movement and curr.left_movement:
                if left_movement_dir is None:
                    left_movement_dir = curr.left_movement
                if curr.left_movement != left_movement_dir:
                    consistent_left = False

            # Check right hand consistency
            if prev.right_movement and curr.right_movement:
                if right_movement_dir is None:
                    right_movement_dir = curr.right_movement
                if curr.right_movement != right_movement_dir:
                    consistent_right = False

            # Check head consistency
            if prev.head_movement and curr.head_movement:
                if head_movement_dir is None:
                    head_movement_dir = curr.head_movement
                if curr.head_movement != head_movement_dir:
                    consistent_head = False

        # DETECT TWO-HANDED GESTURE
        if (consistent_left and consistent_right and
                left_movement_dir and right_movement_dir and
                left_movement_dir != MovementDirection.STILL and
                right_movement_dir != MovementDirection.STILL):

            last_gesture = self.analysis_data['gestures'][-1] if self.analysis_data['gestures'] else None
            if not last_gesture or last_gesture.frame != last_frame.frame:
                gesture = Gesture(
                    type=GestureType.TWO_HANDED_GESTURE,
                    frame=last_frame.frame,
                    time=last_frame.timestamp,
                    left_shape=last_frame.left_handshape,
                    right_shape=last_frame.right_handshape,
                    left_orientation=last_frame.left_orientation,
                    right_orientation=last_frame.right_orientation,
                    left_palm_orientation=last_frame.left_palm_orientation,
                    right_palm_orientation=last_frame.right_palm_orientation,
                    movement=left_movement_dir,
                    description=f"Both hands moving {left_movement_dir.value} - "
                                f"Left: {last_frame.left_handshape.value} ({last_frame.left_orientation.value}), "
                                f"Right: {last_frame.right_handshape.value} ({last_frame.right_orientation.value})"
                )
                self.analysis_data['gestures'].append(gesture)

        # DETECT GESTURE WITH HEAD MOVEMENT
        if ((consistent_left or consistent_right) and
                consistent_head and head_movement_dir and
                head_movement_dir != MovementDirection.STILL):

            is_left_hand = consistent_left
            hand_movement = left_movement_dir if is_left_hand else right_movement_dir
            hand_shape = last_frame.left_handshape if is_left_hand else last_frame.right_handshape
            hand_orientation = last_frame.left_orientation if is_left_hand else last_frame.right_orientation
            palm_orientation = last_frame.left_palm_orientation if is_left_hand else last_frame.right_palm_orientation

            last_gesture = self.analysis_data['gestures'][-1] if self.analysis_data['gestures'] else None
            if not last_gesture or last_gesture.frame != last_frame.frame:
                gesture = Gesture(
                    type=GestureType.GESTURE_WITH_HEAD_MOVEMENT,
                    frame=last_frame.frame,
                    time=last_frame.timestamp,
                    hand_shape=hand_shape,
                    hand_orientation=hand_orientation,
                    palm_orientation=palm_orientation,
                    hand_movement=hand_movement,
                    head_movement=head_movement_dir,
                    description=f"Hand moving {hand_movement.value} ({hand_orientation.value}) "
                                f"with head {head_movement_dir.value}"
                )
                self.analysis_data['gestures'].append(gesture)

        # DETECT HANDSHAPE GESTURE
        if not consistent_left and not consistent_right and not consistent_head:
            if last_frame.left_handshape != HandShape.UNKNOWN:
                last_gesture = self.analysis_data['gestures'][-1] if self.analysis_data['gestures'] else None
                if not last_gesture or last_gesture.frame != last_frame.frame:
                    gesture = Gesture(
                        type=GestureType.HANDSHAPE_GESTURE,
                        frame=last_frame.frame,
                        time=last_frame.timestamp,
                        hand_shape=last_frame.left_handshape,
                        hand_orientation=last_frame.left_orientation,
                        palm_orientation=last_frame.left_palm_orientation,
                        description=f"Handshape: {last_frame.left_handshape.value} "
                                    f"({last_frame.left_orientation.value}, {last_frame.left_palm_orientation.value})"
                    )
                    self.analysis_data['gestures'].append(gesture)

    def generate_sigml(self) -> str:
        """Generate SiGML XML in the correct format for avatar animation"""
        if not self.analysis_data['gestures']:
            warnings.warn("No gestures detected")
            return ""

        # Remove duplicate gestures (same frame)
        unique_gestures = []
        processed_frames = set()

        for gesture in self.analysis_data['gestures']:
            if gesture.frame not in processed_frames:
                unique_gestures.append(gesture)
                processed_frames.add(gesture.frame)

        # Build SiGML document
        root = ET.Element('sigml')

        # Add each gesture as a sign
        self.analysis_data['sigml_signs'] = []

        if unique_gestures:
            for i, gesture in enumerate(unique_gestures[:10], 1):  # Limit to first 10 gestures
                sign_elem = self._create_sigml_sign(gesture, i)
                root.append(sign_elem)
                self.analysis_data['sigml_signs'].append(sign_elem)

        # Convert to string with proper formatting
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')

        # Ensure proper XML declaration
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
        else:
            lines.insert(0, '<?xml version="1.0" encoding="UTF-8"?>')

        return '\n'.join(lines)

    def _create_sigml_sign(self, gesture: Gesture, sign_id: int) -> ET.Element:
        """Create a sign element from detected gesture"""
        gloss = f"sign_{sign_id}"

        # Create sign element
        sign_elem = ET.Element('hns_sign', {'gloss': gloss})

        # Add non-manual elements (head movements)
        if gesture.type == GestureType.GESTURE_WITH_HEAD_MOVEMENT and gesture.head_movement:
            nonmanual_elem = ET.SubElement(sign_elem, 'hamnosys_nonmanual')
            head_elem = self._create_head_movement_element(gesture.head_movement)
            if head_elem is not None:
                nonmanual_elem.append(head_elem)

        # Add manual elements
        manual_elem = ET.SubElement(sign_elem, 'hamnosys_manual')

        # Add two-handed marker if needed
        if gesture.type == GestureType.TWO_HANDED_GESTURE:
            ET.SubElement(manual_elem, 'hamsymmpar')

        # Add handshape
        handshape = (gesture.hand_shape if gesture.hand_shape else
                     gesture.left_shape if gesture.left_shape else
                     HandShape.OPEN)
        handshape_elem = self._create_handshape_element(handshape)
        if handshape_elem is not None:
            manual_elem.append(handshape_elem)

        # Add orientation (finger direction)
        orientation = (gesture.hand_orientation if gesture.hand_orientation else
                       gesture.left_orientation if gesture.left_orientation else
                       HandOrientation.NEUTRAL)
        orientation_elem = self._create_orientation_element(orientation)
        if orientation_elem is not None:
            manual_elem.append(orientation_elem)

        # Add palm orientation
        palm_orientation = (gesture.palm_orientation if gesture.palm_orientation else
                            gesture.left_palm_orientation if gesture.left_palm_orientation else
                            PalmOrientation.PALM_FORWARD)
        palm_elem = self._create_palm_orientation_element(palm_orientation)
        if palm_elem is not None:
            manual_elem.append(palm_elem)

        # Add movement
        movement = (gesture.movement if gesture.movement else
                    gesture.hand_movement if gesture.hand_movement else
                    None)
        if movement and movement != MovementDirection.STILL:
            movement_elem = self._create_movement_element(movement)
            if movement_elem is not None:
                manual_elem.append(movement_elem)

        return sign_elem

    def _create_handshape_element(self, handshape: HandShape) -> Optional[ET.Element]:
        """Create handshape element for SiGML"""
        mapping = {
            HandShape.FIST: 'hamfist',
            HandShape.OPEN: 'hamflathand',
            HandShape.POINT: 'hamfinger2',
            HandShape.VICTORY: 'hamfinger23',
            HandShape.L_SHAPE: 'hamthumboutmod',
            HandShape.OTHER: 'hamflathand',
            HandShape.UNKNOWN: 'hamflathand'
        }
        element_name = mapping.get(handshape)
        if element_name:
            return ET.Element(element_name)
        return None

    def _create_orientation_element(self, orientation: HandOrientation) -> Optional[ET.Element]:
        """Create hand orientation element for SiGML"""
        mapping = {
            HandOrientation.UP: 'hamextfingeru',
            HandOrientation.DOWN: 'hamextfingerd',
            HandOrientation.LEFT: 'hamextfingerl',
            HandOrientation.RIGHT: 'hamextfingerr',
            HandOrientation.NEUTRAL: 'hamextfingeru',
            HandOrientation.UNKNOWN: 'hamextfingeru'
        }
        element_name = mapping.get(orientation)
        if element_name:
            return ET.Element(element_name)
        return None

    def _create_palm_orientation_element(self, palm: PalmOrientation) -> Optional[ET.Element]:
        """Create palm orientation element for SiGML"""
        mapping = {
            PalmOrientation.PALM_UP: 'hampalmu',
            PalmOrientation.PALM_DOWN: 'hampalmd',
            PalmOrientation.PALM_LEFT: 'hampalml',
            PalmOrientation.PALM_RIGHT: 'hampalmr',
            PalmOrientation.PALM_FORWARD: 'hamextfingero',
            PalmOrientation.UNKNOWN: 'hamextfingero'
        }
        element_name = mapping.get(palm)
        if element_name:
            return ET.Element(element_name)
        return None

    def _create_movement_element(self, movement: MovementDirection) -> Optional[ET.Element]:
        """Create movement element for SiGML"""
        mapping = {
            MovementDirection.UP: 'hammoveu',
            MovementDirection.DOWN: 'hammoved',
            MovementDirection.LEFT: 'hammovel',
            MovementDirection.RIGHT: 'hammover',
            MovementDirection.STILL: None,
            MovementDirection.UNKNOWN: None
        }
        element_name = mapping.get(movement)
        if element_name:
            return ET.Element(element_name)
        return None

    def _create_head_movement_element(self, movement: MovementDirection) -> Optional[ET.Element]:
        """Create head movement element for SiGML"""
        mapping = {
            MovementDirection.UP: 'hamheadmovement',
            MovementDirection.DOWN: 'hamheadmovement',
            MovementDirection.LEFT: 'hamheadmovement',
            MovementDirection.RIGHT: 'hamheadmovement',
            MovementDirection.STILL: None,
            MovementDirection.UNKNOWN: None
        }
        element_name = mapping.get(movement)
        if element_name:
            elem = ET.Element(element_name)
            if movement == MovementDirection.UP:
                elem.set('type', 'tilted_backward')
            elif movement == MovementDirection.DOWN:
                elem.set('type', 'tilted_forward')
            elif movement == MovementDirection.LEFT:
                elem.set('type', 'turned_left')
            elif movement == MovementDirection.RIGHT:
                elem.set('type', 'turned_right')
            return elem
        return None

    def close(self):
        """Release resources"""
        self.holistic.close()


# =========================
# FLASK ROUTES
# =========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Save uploaded file temporarily
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)
    
    try:
        # Initialize analyzer
        analyzer = SignLanguageAnalyzer(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process video
        max_frames = request.form.get('max_frames', 300)
        if max_frames:
            max_frames = int(max_frames)
        else:
            max_frames = None
            
        frame_skip = int(request.form.get('frame_skip', 2))
        
        analysis = analyzer.process_video(
            video_path,
            max_frames=max_frames,
            frame_skip=frame_skip
        )
        
        # Generate SiGML
        sigml = analyzer.generate_sigml()
        
        # Prepare response data
        gestures_data = []
        for gesture in analysis['gestures']:
            gesture_dict = {
                'type': gesture.type.value,
                'frame': gesture.frame,
                'time': round(gesture.time, 2),
                'description': gesture.description,
                'hand_shape': gesture.hand_shape.value if gesture.hand_shape else None,
                'hand_orientation': gesture.hand_orientation.value if gesture.hand_orientation else None,
                'palm_orientation': gesture.palm_orientation.value if gesture.palm_orientation else None,
                'movement': gesture.movement.value if gesture.movement else None,
                'head_movement': gesture.head_movement.value if gesture.head_movement else None
            }
            gestures_data.append(gesture_dict)
        
        # Count gesture types
        gesture_counts = {
            'TWO_HANDED_GESTURE': 0,
            'GESTURE_WITH_HEAD_MOVEMENT': 0,
            'HANDSHAPE_GESTURE': 0,
            'total': len(analysis['gestures'])
        }
        
        for gesture in analysis['gestures']:
            if gesture.type.value in gesture_counts:
                gesture_counts[gesture.type.value] += 1
        
        analyzer.close()
        
        # Clean up temp file
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'frames_processed': len(analysis['frames']),
            'gestures_detected': len(analysis['gestures']),
            'gesture_counts': gesture_counts,
            'gestures': gestures_data[:20],  # Return first 20 gestures
            'sigml': sigml
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-sigml', methods=['POST'])
def download_sigml():
    data = request.get_json()
    sigml_content = data.get('sigml', '')
    
    if not sigml_content:
        return jsonify({'error': 'No SiGML content provided'}), 400
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.sigml', delete=False)
    temp_file.write(sigml_content)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name='sign_language_analysis.sigml',
        mimetype='application/xml'
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)   