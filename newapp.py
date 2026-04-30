from flask import Flask, render_template, request, jsonify, send_file
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
import subprocess
import ffmpeg
from PIL import Image
import io

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
    
    # Extended handshapes for HamNoSys
    FLAT_HAND = "flat_hand"
    SPREAD_FINGERS = "spread_fingers"
    PINCH = "pinch"
    C_SHAPE = "c_shape"
    HOOK = "hook"
    BENT = "bent"
    THUMB_ONLY = "thumb_only"
    INDEX_ONLY = "index_only"
    MIDDLE_ONLY = "middle_only"
    RING_ONLY = "ring_only"
    PINKY_ONLY = "pinky_only"
    THUMB_INDEX = "thumb_index"
    THUMB_INDEX_CROSS = "thumb_index_cross"
    THUMB_RING = "thumb_ring"
    DOUBLE_BENT = "double_bent"
    DOUBLE_HOOKED = "double_hooked"


class HandOrientation(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"
    
    # Extended orientations for HamNoSys
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"
    INWARD = "INWARD"
    OUTWARD = "OUTWARD"


class PalmOrientation(Enum):
    PALM_UP = "PALM_UP"
    PALM_DOWN = "PALM_DOWN"
    PALM_LEFT = "PALM_LEFT"
    PALM_RIGHT = "PALM_RIGHT"
    PALM_FORWARD = "PALM_FORWARD"
    UNKNOWN = "UNKNOWN"
    
    # Extended palm orientations
    PALM_UP_LEFT = "PALM_UP_LEFT"
    PALM_UP_RIGHT = "PALM_UP_RIGHT"
    PALM_DOWN_LEFT = "PALM_DOWN_LEFT"
    PALM_DOWN_RIGHT = "PALM_DOWN_RIGHT"
    PALM_BACK = "PALM_BACK"


class MovementDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STILL = "STILL"
    UNKNOWN = "UNKNOWN"
    
    # Extended movements
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"
    CIRCLE = "CIRCLE"
    WAVE = "WAVE"
    REPEATED = "REPEATED"


class Location(Enum):
    HEAD = "head"
    UNDERCHIN = "underchin"
    HANDBACK = "handback"
    NEUTRAL = "neutral"
    UNKNOWN = "UNKNOWN"
    
    # Extended locations from HamNoSys
    HEAD_TOP = "head_top"
    FOREHEAD = "forehead"
    NOSE = "nose"
    NOSTRILS = "nostrils"
    LIPS = "lips"
    TONGUE = "tongue"
    TEETH = "teeth"
    CHIN = "chin"
    NECK = "neck"
    SHOULDER_TOP = "shoulder_top"
    SHOULDERS = "shoulders"
    CHEST = "chest"
    STOMACH = "stomach"
    BELOW_STOMACH = "below_stomach"
    EYEBROWS = "eyebrows"
    EYES = "eyes"
    EAR = "ear"
    EARLOBE = "earlobe"
    CHEEK = "cheek"
    PALM = "palm"
    THUMB_SIDE = "thumb_side"
    TOUCH = "touch"
    FOREARM = "forearm"


class GestureType(Enum):
    TWO_HANDED_GESTURE = "TWO_HANDED_GESTURE"
    GESTURE_WITH_HEAD_MOVEMENT = "GESTURE_WITH_HEAD_MOVEMENT"
    HANDSHAPE_GESTURE = "HANDSHAPE_GESTURE"
    MANUAL_GESTURE = "MANUAL_GESTURE"
    LOCATION_GESTURE = "LOCATION_GESTURE"
    SYMMETRIC_TWO_HANDED = "SYMMETRIC_TWO_HANDED"
    ALTERNATING_TWO_HANDED = "ALTERNATING_TWO_HANDED"
    SEQUENTIAL_TWO_HANDED = "SEQUENTIAL_TWO_HANDED"
    PARALLEL_TWO_HANDED = "PARALLEL_TWO_HANDED"


class TwoHandedType(Enum):
    SYMMETRIC = "symmetric"
    MIRROR = "mirror"
    ALTERNATING = "alternating"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FUSION = "fusion"
    RELATIVE = "relative"


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
    two_handed_type: Optional[TwoHandedType] = None


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
    two_handed_type: Optional[TwoHandedType] = None
    confidence: float = 1.0
    description: str = ""
    sign_name: Optional[str] = None


# =========================
# HAMNOSYS MAPPING CLASSES
# =========================

class HamNoSysMapper:
    """Maps detected gestures to HamNoSys notation elements"""
    
    # Handshape mappings
    HANDSHAPE_MAP = {
        # Basic handshapes
        HandShape.FIST: 'hamfist',
        HandShape.OPEN: 'hamflathand',
        HandShape.POINT: 'hamfinger2',
        HandShape.VICTORY: 'hamfinger23spread',
        HandShape.L_SHAPE: 'hamthumbopenmod',
        HandShape.FLAT_HAND: 'hamflathand',
        HandShape.SPREAD_FINGERS: 'hamfinger2345',
        HandShape.PINCH: 'hampinch12',
        HandShape.C_SHAPE: 'hamceeall',
        HandShape.HOOK: 'hamfingerhookmod',
        HandShape.BENT: 'hamfingerbendmod',
        HandShape.THUMB_ONLY: 'hamthumb',
        HandShape.INDEX_ONLY: 'hamindexfinger',
        HandShape.MIDDLE_ONLY: 'hammiddlefinger',
        HandShape.RING_ONLY: 'hamringfinger',
        HandShape.PINKY_ONLY: 'hampinky',
        HandShape.THUMB_INDEX: 'hambetween',
        HandShape.THUMB_INDEX_CROSS: 'hamfingertip',
        HandShape.THUMB_RING: 'hamfingermidjoint',
        HandShape.DOUBLE_BENT: 'hamdoublebent',
        HandShape.DOUBLE_HOOKED: 'hamdoublehooked',
        HandShape.OTHER: 'hamflathand',
        HandShape.UNKNOWN: 'hamflathand'
    }
    
    # Orientation mappings (finger direction)
    ORIENTATION_MAP = {
        HandOrientation.UP: 'hamextfingeru',
        HandOrientation.UP_LEFT: 'hamextfingerul',
        HandOrientation.UP_RIGHT: 'hamextfingerur',
        HandOrientation.DOWN: 'hamextfingerd',
        HandOrientation.DOWN_LEFT: 'hamextfingerdl',
        HandOrientation.DOWN_RIGHT: 'hamextfingerdr',
        HandOrientation.LEFT: 'hamextfingerl',
        HandOrientation.RIGHT: 'hamextfingerr',
        HandOrientation.INWARD: 'hamextfingeri',
        HandOrientation.OUTWARD: 'hamextfingero',
        HandOrientation.NEUTRAL: 'hamextfingero',
        HandOrientation.UNKNOWN: 'hamextfingero'
    }
    
    # Extended orientation variations
    ORIENTATION_VARIATIONS = {
        ('UP', 'slight_left'): 'hamextfingerul',
        ('UP', 'slight_right'): 'hamextfingerur',
        ('UP', 'touch'): 'hamextfingeru',
        ('UP', 'inward'): 'hamextfingerui',
        ('UP', 'outward'): 'hamextfingeruo',
        ('DOWN', 'slight_left'): 'hamextfingerdl',
        ('DOWN', 'slight_right'): 'hamextfingerdr',
        ('DOWN', 'outward'): 'hamextfingerdo',
        ('LEFT', 'touch'): 'hamextfingerl',
        ('RIGHT', 'touch'): 'hamextfingerr',
        ('INWARD', 'middle'): 'hamextfingeri',
        ('OUTWARD', 'forward'): 'hamextfingero',
        ('OUTWARD', 'left'): 'hamextfingerol',
        ('OUTWARD', 'right'): 'hamextfingeror'
    }
    
    # Palm orientation mappings
    PALM_MAP = {
        PalmOrientation.PALM_UP: 'hampalmu',
        PalmOrientation.PALM_UP_LEFT: 'hampalmul',
        PalmOrientation.PALM_UP_RIGHT: 'hampalmur',
        PalmOrientation.PALM_DOWN: 'hampalmd',
        PalmOrientation.PALM_DOWN_LEFT: 'hampalmdl',
        PalmOrientation.PALM_DOWN_RIGHT: 'hampalmdr',
        PalmOrientation.PALM_LEFT: 'hampalml',
        PalmOrientation.PALM_RIGHT: 'hampalmr',
        PalmOrientation.PALM_FORWARD: 'hamextfingero',
        PalmOrientation.PALM_BACK: 'hamextfingeri',
        PalmOrientation.UNKNOWN: 'hamextfingero'
    }
    
    # Palm orientation variations
    PALM_VARIATIONS = {
        ('PALM_UP', 'neutral'): 'hampalmu',
        ('PALM_UP', 'left_tilt'): 'hampalmul',
        ('PALM_UP', 'right_tilt'): 'hampalmur',
        ('PALM_UP', 'down_tilt'): 'hampalmud',
        ('PALM_UP', 'down_left_tilt'): 'hampalmudl',
        ('PALM_UP', 'down_right_tilt'): 'hampalmudr',
        ('PALM_DOWN', 'neutral'): 'hampalmd',
        ('PALM_DOWN', 'left_tilt'): 'hampalmdl',
        ('PALM_DOWN', 'right_tilt'): 'hampalmdr',
    }
    
    # Movement mappings
    MOVEMENT_MAP = {
        MovementDirection.UP: 'hammoveu',
        MovementDirection.UP_LEFT: 'hammoveul',
        MovementDirection.UP_RIGHT: 'hammoveur',
        MovementDirection.DOWN: 'hammoved',
        MovementDirection.DOWN_LEFT: 'hammovedl',
        MovementDirection.DOWN_RIGHT: 'hammovedr',
        MovementDirection.LEFT: 'hammovel',
        MovementDirection.RIGHT: 'hammover',
        MovementDirection.CIRCLE: 'hamcirclemove',
        MovementDirection.WAVE: 'hamwavemove',
        MovementDirection.REPEATED: 'hamrepeatedmove',
        MovementDirection.STILL: None,
        MovementDirection.UNKNOWN: None
    }
    
    # Location mappings
    LOCATION_MAP = {
        Location.HEAD_TOP: 'hamheadtop',
        Location.HEAD: 'hamhead',
        Location.FOREHEAD: 'hamforehead',
        Location.NOSE: 'hamnose',
        Location.NOSTRILS: 'hamnostrils',
        Location.LIPS: 'hamlips',
        Location.TONGUE: 'hamtongue',
        Location.TEETH: 'hamteeth',
        Location.CHIN: 'hamchin',
        Location.UNDERCHIN: 'hamunderchin',
        Location.NECK: 'hamneck',
        Location.SHOULDER_TOP: 'hamshouldertop',
        Location.SHOULDERS: 'hamshoulders',
        Location.CHEST: 'hamchest',
        Location.STOMACH: 'hamstomach',
        Location.BELOW_STOMACH: 'hambelowstomach',
        Location.EYEBROWS: 'hameyebrows',
        Location.EYES: 'hameyes',
        Location.EAR: 'hamear',
        Location.EARLOBE: 'hamearlobe',
        Location.CHEEK: 'hamcheek',
        Location.PALM: 'hampalm',
        Location.HANDBACK: 'hamhandback',
        Location.THUMB_SIDE: 'hamthumbside',
        Location.TOUCH: 'hamtouch',
        Location.FOREARM: 'hamforearm',
        Location.NEUTRAL: 'hamneutral',
        Location.UNKNOWN: None
    }
    
    # Two-handed gesture types
    TWO_HANDED_MAP = {
        TwoHandedType.SYMMETRIC: 'hamsymmpar',
        TwoHandedType.MIRROR: 'hamsymmir',
        TwoHandedType.ALTERNATING: 'hamaltbegin',
        TwoHandedType.SEQUENTIAL: 'hamseqbegin',
        TwoHandedType.PARALLEL: 'hamparbegin',
        TwoHandedType.FUSION: 'hamfusionbegin',
        TwoHandedType.RELATIVE: 'hamorirelative'
    }
    
    # Head movement mappings
    HEAD_MOVEMENT_MAP = {
        MovementDirection.UP: ('hamheadmovement', 'tilted_backward'),
        MovementDirection.DOWN: ('hamheadmovement', 'tilted_forward'),
        MovementDirection.LEFT: ('hamheadmovement', 'turned_left'),
        MovementDirection.RIGHT: ('hamheadmovement', 'turned_right'),
        MovementDirection.UP_LEFT: ('hamheadmovement', 'tilted_backward_left'),
        MovementDirection.UP_RIGHT: ('hamheadmovement', 'tilted_backward_right'),
        MovementDirection.DOWN_LEFT: ('hamheadmovement', 'tilted_forward_left'),
        MovementDirection.DOWN_RIGHT: ('hamheadmovement', 'tilted_forward_right'),
    }
    
    @classmethod
    def get_handshape_element(cls, handshape: HandShape, variation: str = None) -> Optional[ET.Element]:
        """Get handshape element with optional variation"""
        element_name = cls.HANDSHAPE_MAP.get(handshape)
        if element_name:
            elem = ET.Element(element_name)
            if variation:
                elem.set('variation', variation)
            return elem
        return None
    
    @classmethod
    def get_orientation_element(cls, orientation: HandOrientation, 
                               variation: str = None, touch: bool = False) -> Optional[ET.Element]:
        """Get orientation element with variations"""
        if variation and (orientation, variation) in cls.ORIENTATION_VARIATIONS:
            element_name = cls.ORIENTATION_VARIATIONS[(orientation, variation)]
        else:
            element_name = cls.ORIENTATION_MAP.get(orientation)
        
        if element_name:
            elem = ET.Element(element_name)
            if touch:
                elem.set('touch', 'true')
            return elem
        return None
    
    @classmethod
    def get_palm_element(cls, palm: PalmOrientation, tilt: str = None) -> Optional[ET.Element]:
        """Get palm orientation element with tilt variations"""
        if tilt and (palm, tilt) in cls.PALM_VARIATIONS:
            element_name = cls.PALM_VARIATIONS[(palm, tilt)]
        else:
            element_name = cls.PALM_MAP.get(palm)
        
        if element_name:
            return ET.Element(element_name)
        return None
    
    @classmethod
    def get_movement_element(cls, movement: MovementDirection, 
                            repeated: bool = False) -> Optional[ET.Element]:
        """Get movement element"""
        element_name = cls.MOVEMENT_MAP.get(movement)
        if element_name:
            elem = ET.Element(element_name)
            if repeated:
                elem.set('repeated', 'true')
            return elem
        return None
    
    @classmethod
    def get_location_element(cls, location: Location) -> Optional[ET.Element]:
        """Get location element"""
        element_name = cls.LOCATION_MAP.get(location)
        if element_name:
            return ET.Element(element_name)
        return None
    
    @classmethod
    def get_two_handed_element(cls, two_handed_type: TwoHandedType) -> Optional[ET.Element]:
        """Get two-handed gesture element"""
        element_name = cls.TWO_HANDED_MAP.get(two_handed_type)
        if element_name:
            return ET.Element(element_name)
        return None
    
    @classmethod
    def get_head_movement_element(cls, movement: MovementDirection) -> Optional[ET.Element]:
        """Get head movement element with type attribute"""
        if movement in cls.HEAD_MOVEMENT_MAP:
            element_name, move_type = cls.HEAD_MOVEMENT_MAP[movement]
            elem = ET.Element(element_name)
            elem.set('type', move_type)
            return elem
        return None


# =========================
# SIGN LANGUAGE ANALYZER CLASS (FFMPEG Version - No OpenCV)
# =========================

class SignLanguageAnalyzer:
    """Main class for sign language analysis from video using FFmpeg (no OpenCV)"""

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
        
        # HamNoSys mapper
        self.hamnosys_mapper = HamNoSysMapper()

    def get_video_info(self, video_path: str) -> Tuple[float, int, int, int]:
        """Get video FPS, total frame count, width, and height using ffmpeg"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Get FPS
            fps = eval(video_info['r_frame_rate'])
            
            # Get dimensions
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Get total frames
            total_frames = int(video_info.get('nb_frames', 0))
            if total_frames == 0:
                # Calculate from duration if nb_frames not available
                duration = float(video_info['duration'])
                total_frames = int(duration * fps)
            
            return fps, total_frames, width, height
        except Exception as e:
            print(f"Error getting video info: {e}")
            return 30.0, 0, 640, 480  # Default fallback

    def extract_frames_ffmpeg(self, video_path: str, max_frames: Optional[int] = None, 
                              frame_skip: int = 1) -> Tuple[List[np.ndarray], List[float], float]:
        """
        Extract frames from video using FFmpeg directly to RGB format
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            frame_skip: Process every Nth frame
        
        Returns:
            Tuple of (frames list, timestamps list, fps)
        """
        frames = []
        timestamps = []
        
        # Get video info
        fps, total_frames, width, height = self.get_video_info(video_path)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Calculate target frame rate for extraction
        target_fps = fps / frame_skip
        
        try:
            # Use ffmpeg to extract frames as RGB images
            out, err = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=target_fps)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Convert raw video data to numpy arrays
            frame_size = width * height * 3  # 3 bytes per pixel for RGB
            
            frame_count = 0
            for i in range(0, len(out), frame_size):
                if max_frames and frame_count >= total_frames:
                    break
                    
                # Extract frame data
                frame_data = out[i:i + frame_size]
                if len(frame_data) < frame_size:
                    break
                
                # Convert to numpy array and reshape - this is RGB already
                frame = np.frombuffer(frame_data, np.uint8).reshape((height, width, 3))
                frames.append(frame)
                
                # Calculate timestamp
                timestamp = frame_count / target_fps
                timestamps.append(timestamp)
                
                frame_count += 1
                
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise
        
        return frames, timestamps, target_fps

    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                      frame_skip: int = 1, callback=None):
        """
        Process a video file using FFmpeg and detect sign language gestures
        """
        # Extract frames using FFmpeg
        frames, timestamps, fps = self.extract_frames_ffmpeg(
            video_path, 
            max_frames=max_frames, 
            frame_skip=frame_skip
        )
        
        self.analysis_data = {'frames': [], 'gestures': [], 'sigml_signs': []}
        self.gesture_buffer = []
        self.previous_frame_data = None
        self.left_handshape_history = []
        self.right_handshape_history = []
        self.location_history = []
        
        processed_count = 0
        
        for frame_count, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Process frame with MediaPipe - frame is already RGB
            results = self.holistic.process(frame)
            frame_data = self._process_frame_data(results, processed_count, timestamp)
            
            self.analysis_data['frames'].append(frame_data)
            self._detect_gestures_with_buffer(frame_data)
            
            processed_count += 1
            
            if callback and processed_count % 10 == 0:
                progress = (frame_count / len(frames)) * 100
                callback(progress, frame_data)
        
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
            data.left_location = self._detect_location(data.left_hand, data.nose_pos)

        # Process right hand
        if results.right_hand_landmarks:
            data.right_hand = self._extract_landmarks(results.right_hand_landmarks.landmark)
            data.right_handshape = self._get_handshape(data.right_hand)
            data.right_orientation = self._get_hand_orientation(data.right_hand)
            data.right_palm_orientation = self._get_palm_orientation(data.right_hand)
            data.right_wrist_pos = data.right_hand[0]
            data.right_location = self._detect_location(data.right_hand, data.nose_pos)

        # Process pose
        if results.pose_landmarks:
            data.head = self._extract_landmarks(results.pose_landmarks.landmark)
            data.head_orientation = self._get_head_orientation(data.head)
            data.nose_pos = data.head[0]

        # Detect two-handed gesture type if both hands present
        if data.left_hand and data.right_hand:
            data.two_handed_type = self._detect_two_handed_type(data)

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
        diagonal_threshold = 0.2

        # Check for diagonal directions first
        if abs(norm_dx) > diagonal_threshold and abs(norm_dy) > diagonal_threshold:
            if norm_dy < -threshold and norm_dx < -threshold:
                return HandOrientation.UP_LEFT
            elif norm_dy < -threshold and norm_dx > threshold:
                return HandOrientation.UP_RIGHT
            elif norm_dy > threshold and norm_dx < -threshold:
                return HandOrientation.DOWN_LEFT
            elif norm_dy > threshold and norm_dx > threshold:
                return HandOrientation.DOWN_RIGHT

        # Check cardinal directions
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

        # Check for inward/outward (using z-coordinate)
        if hasattr(wrist, 'z') and hasattr(middle_tip, 'z'):
            dz = middle_tip.z - wrist.z
            if abs(dz) > 0.01:
                return HandOrientation.INWARD if dz < 0 else HandOrientation.OUTWARD

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

        # Check for diagonal palm orientations
        if abs(norm_dy) > 0.2 and abs(norm_dx) > 0.2:
            if norm_dy < -threshold and norm_dx < -threshold:
                return PalmOrientation.PALM_UP_LEFT
            elif norm_dy < -threshold and norm_dx > threshold:
                return PalmOrientation.PALM_UP_RIGHT
            elif norm_dy > threshold and norm_dx < -threshold:
                return PalmOrientation.PALM_DOWN_LEFT
            elif norm_dy > threshold and norm_dx > threshold:
                return PalmOrientation.PALM_DOWN_RIGHT

        # Check cardinal orientations
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

        # Check for forward/back (using z-coordinate of palm center)
        palm_center_z = (index_mcp.z + pinky_mcp.z) / 2
        if palm_center_z < wrist.z - 0.02:
            return PalmOrientation.PALM_FORWARD
        elif palm_center_z > wrist.z + 0.02:
            return PalmOrientation.PALM_BACK

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

        # Check for diagonal movements
        if abs(dx) > threshold and abs(dy) > threshold:
            if dx > 0 and dy < 0:
                return MovementDirection.UP_RIGHT
            elif dx > 0 and dy > 0:
                return MovementDirection.DOWN_RIGHT
            elif dx < 0 and dy < 0:
                return MovementDirection.UP_LEFT
            elif dx < 0 and dy > 0:
                return MovementDirection.DOWN_LEFT

        # Check cardinal directions
        if abs(dx) > abs(dy):
            return MovementDirection.RIGHT if dx > 0 else MovementDirection.LEFT
        return MovementDirection.DOWN if dy > 0 else MovementDirection.UP

    def _detect_location(self, hand_landmarks: List[Landmark], 
                         nose_pos: Optional[Landmark]) -> Location:
        """Detect hand location relative to face and body"""
        if not hand_landmarks or not nose_pos:
            return Location.UNKNOWN

        wrist = hand_landmarks[0]
        
        # Calculate relative position to nose
        rel_y = wrist.y - nose_pos.y
        rel_x = wrist.x - nose_pos.x
        
        # Head locations
        if abs(rel_x) < 0.1:  # Centered on face
            if rel_y < -0.1:
                return Location.HEAD_TOP
            elif rel_y < 0:
                return Location.FOREHEAD
            elif rel_y < 0.1:
                return Location.NOSE
            elif rel_y < 0.15:
                return Location.LIPS
            elif rel_y < 0.2:
                return Location.CHIN
            elif rel_y < 0.3:
                return Location.UNDERCHIN
        elif rel_x > 0.1:  # Right side of face
            if abs(rel_y) < 0.1:
                return Location.EAR
            elif rel_y < 0 and rel_y > -0.1:
                return Location.EYE
            elif rel_y < 0.1:
                return Location.CHEEK
        elif rel_x < -0.1:  # Left side of face
            if abs(rel_y) < 0.1:
                return Location.EAR
            elif rel_y < 0 and rel_y > -0.1:
                return Location.EYE
            elif rel_y < 0.1:
                return Location.CHEEK
        
        # Body locations
        if rel_y > 0.2 and rel_y < 0.4:
            return Location.NECK
        elif rel_y > 0.4 and rel_y < 0.6:
            return Location.SHOULDERS
        elif rel_y > 0.6 and rel_y < 0.8:
            return Location.CHEST
        elif rel_y > 0.8:
            return Location.STOMACH
            
        return Location.NEUTRAL

    def _detect_two_handed_type(self, frame_data: FrameData) -> TwoHandedType:
        """Detect the type of two-handed gesture"""
        if not frame_data.left_hand or not frame_data.right_hand:
            return None
            
        left_wrist = frame_data.left_wrist_pos
        right_wrist = frame_data.right_wrist_pos
        
        if not left_wrist or not right_wrist:
            return None
            
        # Check if hands are moving symmetrically
        if (frame_data.left_movement == frame_data.right_movement and
            frame_data.left_movement != MovementDirection.STILL):
            # Check if movements are mirror (opposite x directions)
            if ((frame_data.left_movement in [MovementDirection.LEFT, MovementDirection.RIGHT]) and
                frame_data.left_movement != frame_data.right_movement):
                return TwoHandedType.MIRROR
            else:
                return TwoHandedType.SYMMETRIC
                
        # Check for alternating movement
        if (frame_data.left_movement and frame_data.right_movement and
            frame_data.left_movement != frame_data.right_movement):
            return TwoHandedType.ALTERNATING
            
        # Check hand positions relative to each other
        distance = math.sqrt((left_wrist.x - right_wrist.x)**2 + 
                           (left_wrist.y - right_wrist.y)**2)
        
        if distance < 0.05:  # Hands close together
            return TwoHandedType.FUSION
        elif distance > 0.3:  # Hands far apart
            return TwoHandedType.PARALLEL
        elif abs(left_wrist.y - right_wrist.y) < 0.02:  # Same height
            return TwoHandedType.RELATIVE
            
        return TwoHandedType.SYMMETRIC

    def _get_handshape(self, landmarks: List[Landmark]) -> HandShape:
        """Determine handshape from hand landmarks with extended classifications"""
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
        
        # Calculate finger bend states
        bend_states = {
            'thumb': self._is_finger_bent(landmarks, [0, 2, 3, 4]),
            'index': self._is_finger_bent(landmarks, [0, 5, 6, 8]),
            'middle': self._is_finger_bent(landmarks, [0, 9, 10, 12]),
            'ring': self._is_finger_bent(landmarks, [0, 13, 14, 16]),
            'pinky': self._is_finger_bent(landmarks, [0, 17, 18, 20])
        }
        
        # Calculate finger spread
        spread = self._calculate_finger_spread(landmarks)

        # Count extended fingers
        extended_count = sum(finger_states.values())
        
        # Check for specific handshapes
        
        # Single finger extensions
        if extended_count == 1:
            if finger_states['thumb']:
                return HandShape.THUMB_ONLY
            elif finger_states['index']:
                return HandShape.INDEX_ONLY
            elif finger_states['middle']:
                return HandShape.MIDDLE_ONLY
            elif finger_states['ring']:
                return HandShape.RING_ONLY
            elif finger_states['pinky']:
                return HandShape.PINKY_ONLY
                
        # Two finger combinations
        if extended_count == 2:
            if finger_states['thumb'] and finger_states['index']:
                # Check if it's pinch or L-shape
                if self._is_pinch(landmarks):
                    return HandShape.PINCH
                elif spread > 0.03:
                    return HandShape.L_SHAPE
                else:
                    return HandShape.THUMB_INDEX
            elif finger_states['index'] and finger_states['middle']:
                if spread > 0.03:
                    return HandShape.VICTORY
                else:
                    return HandShape.OTHER
            elif finger_states['thumb'] and finger_states['ring']:
                return HandShape.THUMB_RING
                
        # Three finger extensions
        if extended_count == 3:
            if finger_states['index'] and finger_states['middle'] and finger_states['thumb']:
                return HandShape.POINT
                
        # Four or more fingers
        if extended_count >= 4:
            if all(bend_states.values()):
                return HandShape.BENT
            elif spread > 0.05:
                return HandShape.SPREAD_FINGERS
            else:
                return HandShape.OPEN
                
        # Check for specific shapes
        if extended_count == 0:
            if all(bend_states.values()):
                return HandShape.DOUBLE_BENT
            else:
                return HandShape.FIST
                
        # Check for C shape
        if self._is_c_shape(landmarks):
            return HandShape.C_SHAPE
            
        # Check for hook
        if self._is_hook(landmarks):
            return HandShape.HOOK

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

    def _is_finger_bent(self, landmarks: List[Landmark], indices: List[int]) -> bool:
        """Check if a finger is bent"""
        try:
            mcp, pip, tip = [landmarks[i] for i in indices[1:4]]
            angle = self._calculate_angle(mcp, pip, tip)
            return angle < 120
        except (IndexError, AttributeError):
            return False

    def _is_pinch(self, landmarks: List[Landmark]) -> bool:
        """Check if hand is making a pinch gesture"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                               (thumb_tip.y - index_tip.y)**2)
            
            return distance < 0.02
        except (IndexError, AttributeError):
            return False

    def _is_c_shape(self, landmarks: List[Landmark]) -> bool:
        """Check if hand is making a C shape"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinky_tip = landmarks[20]
            
            # Check if fingers are curved
            thumb_curved = self._is_finger_bent(landmarks, [0, 2, 3, 4])
            index_curved = self._is_finger_bent(landmarks, [0, 5, 6, 8])
            
            # Check spread
            spread = abs(index_tip.x - pinky_tip.x)
            
            return thumb_curved and index_curved and spread > 0.03
        except (IndexError, AttributeError):
            return False

    def _is_hook(self, landmarks: List[Landmark]) -> bool:
        """Check if fingers are in hook position"""
        try:
            hook_count = 0
            for i in [8, 12, 16, 20]:  # Finger tips
                tip = landmarks[i]
                pip = landmarks[i-2]
                if tip.y > pip.y + 0.02:  # Tip below PIP (pointing down)
                    hook_count += 1
            return hook_count >= 3
        except (IndexError, AttributeError):
            return False

    def _calculate_finger_spread(self, landmarks: List[Landmark]) -> float:
        """Calculate average spread between fingers"""
        try:
            spreads = []
            finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
            
            for i in range(len(finger_tips)-1):
                tip1 = landmarks[finger_tips[i]]
                tip2 = landmarks[finger_tips[i+1]]
                spread = abs(tip1.x - tip2.x)
                spreads.append(spread)
                
            return np.mean(spreads) if spreads else 0
        except (IndexError, AttributeError):
            return 0

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
                
                # Determine two-handed type
                two_handed_type = self._detect_two_handed_type(last_frame)
                
                gesture_type = GestureType.TWO_HANDED_GESTURE
                if two_handed_type == TwoHandedType.SYMMETRIC:
                    gesture_type = GestureType.SYMMETRIC_TWO_HANDED
                elif two_handed_type == TwoHandedType.ALTERNATING:
                    gesture_type = GestureType.ALTERNATING_TWO_HANDED
                
                gesture = Gesture(
                    type=gesture_type,
                    frame=last_frame.frame,
                    time=last_frame.timestamp,
                    left_shape=last_frame.left_handshape,
                    right_shape=last_frame.right_handshape,
                    left_orientation=last_frame.left_orientation,
                    right_orientation=last_frame.right_orientation,
                    left_palm_orientation=last_frame.left_palm_orientation,
                    right_palm_orientation=last_frame.right_palm_orientation,
                    movement=left_movement_dir,
                    two_handed_type=two_handed_type,
                    description=f"Two-handed {two_handed_type.value if two_handed_type else 'gesture'}: "
                                f"Both hands moving {left_movement_dir.value} - "
                                f"Left: {last_frame.left_handshape.value}, "
                                f"Right: {last_frame.right_handshape.value}"
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
                        location=last_frame.left_location,
                        description=f"Handshape: {last_frame.left_handshape.value} "
                                    f"({last_frame.left_orientation.value}, {last_frame.left_palm_orientation.value})"
                                    f" at {last_frame.left_location.value}"
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
            for i, gesture in enumerate(unique_gestures[:15], 1):  # Limit to first 15 gestures
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
        """Create a sign element from detected gesture using HamNoSys mappings"""
        gloss = gesture.sign_name if gesture.sign_name else f"sign_{sign_id}"

        # Create sign element
        sign_elem = ET.Element('hns_sign', {'gloss': gloss})

        # Add non-manual elements (head movements)
        if gesture.type == GestureType.GESTURE_WITH_HEAD_MOVEMENT and gesture.head_movement:
            nonmanual_elem = ET.SubElement(sign_elem, 'hamnosys_nonmanual')
            head_elem = self.hamnosys_mapper.get_head_movement_element(gesture.head_movement)
            if head_elem is not None:
                nonmanual_elem.append(head_elem)

        # Add location elements
        if gesture.location and gesture.location != Location.UNKNOWN:
            location_elem = ET.SubElement(sign_elem, 'hamnosys_location')
            loc_elem = self.hamnosys_mapper.get_location_element(gesture.location)
            if loc_elem is not None:
                location_elem.append(loc_elem)

        # Add manual elements
        manual_elem = ET.SubElement(sign_elem, 'hamnosys_manual')

        # Add two-handed marker if needed
        if gesture.type in [GestureType.TWO_HANDED_GESTURE, 
                           GestureType.SYMMETRIC_TWO_HANDED,
                           GestureType.ALTERNATING_TWO_HANDED] or gesture.two_handed_type:
            
            if gesture.two_handed_type:
                two_handed_elem = self.hamnosys_mapper.get_two_handed_element(gesture.two_handed_type)
                if two_handed_elem is not None:
                    manual_elem.append(two_handed_elem)
            else:
                ET.SubElement(manual_elem, 'hamsymmpar')

        # Add handshape for left/dominant hand
        handshape = (gesture.hand_shape if gesture.hand_shape else
                     gesture.left_shape if gesture.left_shape else
                     gesture.right_shape if gesture.right_shape else
                     HandShape.OPEN)
        
        handshape_elem = self.hamnosys_mapper.get_handshape_element(handshape)
        if handshape_elem is not None:
            manual_elem.append(handshape_elem)

        # Add orientation (finger direction) for left hand
        orientation = (gesture.hand_orientation if gesture.hand_orientation else
                       gesture.left_orientation if gesture.left_orientation else
                       gesture.right_orientation if gesture.right_orientation else
                       HandOrientation.NEUTRAL)
        
        orientation_elem = self.hamnosys_mapper.get_orientation_element(orientation)
        if orientation_elem is not None:
            manual_elem.append(orientation_elem)

        # Add palm orientation for left hand
        palm_orientation = (gesture.palm_orientation if gesture.palm_orientation else
                            gesture.left_palm_orientation if gesture.left_palm_orientation else
                            gesture.right_palm_orientation if gesture.right_palm_orientation else
                            PalmOrientation.PALM_FORWARD)
        
        palm_elem = self.hamnosys_mapper.get_palm_element(palm_orientation)
        if palm_elem is not None:
            manual_elem.append(palm_elem)

        # Add movement
        movement = (gesture.movement if gesture.movement else
                    gesture.hand_movement if gesture.hand_movement else
                    None)
        
        if movement and movement != MovementDirection.STILL:
            movement_elem = self.hamnosys_mapper.get_movement_element(movement)
            if movement_elem is not None:
                manual_elem.append(movement_elem)

        # If two-handed, add right hand elements
        if gesture.type in [GestureType.TWO_HANDED_GESTURE, 
                           GestureType.SYMMETRIC_TWO_HANDED] and gesture.right_shape:
            
            # Add second hand elements
            if gesture.right_shape and gesture.right_shape != HandShape.UNKNOWN:
                right_handshape_elem = self.hamnosys_mapper.get_handshape_element(gesture.right_shape)
                if right_handshape_elem is not None:
                    # Add as separate hand in two-handed context
                    right_hand_elem = ET.SubElement(manual_elem, 'hamnosys_hand', {'side': 'right'})
                    right_hand_elem.append(right_handshape_elem)
                    
                    if gesture.right_orientation:
                        right_orient_elem = self.hamnosys_mapper.get_orientation_element(gesture.right_orientation)
                        if right_orient_elem:
                            right_hand_elem.append(right_orient_elem)
                    
                    if gesture.right_palm_orientation:
                        right_palm_elem = self.hamnosys_mapper.get_palm_element(gesture.right_palm_orientation)
                        if right_palm_elem:
                            right_hand_elem.append(right_palm_elem)

        return sign_elem

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
        
        # Prepare response data with enhanced gesture information
        gestures_data = []
        for gesture in analysis['gestures']:
            gesture_dict = {
                'type': gesture.type.value,
                'frame': gesture.frame,
                'time': round(gesture.time, 2),
                'description': gesture.description,
                'hand_shape': gesture.hand_shape.value if gesture.hand_shape else None,
                'left_shape': gesture.left_shape.value if gesture.left_shape else None,
                'right_shape': gesture.right_shape.value if gesture.right_shape else None,
                'hand_orientation': gesture.hand_orientation.value if gesture.hand_orientation else None,
                'left_orientation': gesture.left_orientation.value if gesture.left_orientation else None,
                'right_orientation': gesture.right_orientation.value if gesture.right_orientation else None,
                'palm_orientation': gesture.palm_orientation.value if gesture.palm_orientation else None,
                'left_palm_orientation': gesture.left_palm_orientation.value if gesture.left_palm_orientation else None,
                'right_palm_orientation': gesture.right_palm_orientation.value if gesture.right_palm_orientation else None,
                'movement': gesture.movement.value if gesture.movement else None,
                'hand_movement': gesture.hand_movement.value if gesture.hand_movement else None,
                'head_movement': gesture.head_movement.value if gesture.head_movement else None,
                'location': gesture.location.value if gesture.location else None,
                'two_handed_type': gesture.two_handed_type.value if gesture.two_handed_type else None,
                'confidence': gesture.confidence
            }
            gestures_data.append(gesture_dict)
        
        # Count gesture types with enhanced categories
        gesture_counts = {
            'TWO_HANDED_GESTURE': 0,
            'SYMMETRIC_TWO_HANDED': 0,
            'ALTERNATING_TWO_HANDED': 0,
            'GESTURE_WITH_HEAD_MOVEMENT': 0,
            'HANDSHAPE_GESTURE': 0,
            'LOCATION_GESTURE': 0,
            'total': len(analysis['gestures'])
        }
        
        for gesture in analysis['gestures']:
            if gesture.type.value in gesture_counts:
                gesture_counts[gesture.type.value] += 1
            if gesture.location and gesture.location != Location.UNKNOWN:
                gesture_counts['LOCATION_GESTURE'] += 1
        
        analyzer.close()
        
        # Clean up temp file
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'frames_processed': len(analysis['frames']),
            'gestures_detected': len(analysis['gestures']),
            'gesture_counts': gesture_counts,
            'gestures': gestures_data[:25],  # Return first 25 gestures
            'sigml': sigml,
            'hamnosys_version': '3.0'
        })
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        return jsonify({'error': f'FFmpeg error: {error_message}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file if it still exists
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass


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


@app.route('/handshapes', methods=['GET'])
def get_handshapes():
    """Return list of supported handshapes"""
    handshapes = [
        {'name': 'FIST', 'hamnosys': 'hamfist', 'description': 'Closed fist'},
        {'name': 'OPEN', 'hamnosys': 'hamflathand', 'description': 'Open flat hand'},
        {'name': 'POINT', 'hamnosys': 'hamfinger2', 'description': 'Pointing index finger'},
        {'name': 'VICTORY', 'hamnosys': 'hamfinger23spread', 'description': 'Victory sign (V)'},
        {'name': 'L_SHAPE', 'hamnosys': 'hamthumbopenmod', 'description': 'L shape with thumb and index'},
        {'name': 'SPREAD_FINGERS', 'hamnosys': 'hamfinger2345', 'description': 'All fingers spread'},
        {'name': 'PINCH', 'hamnosys': 'hampinch12', 'description': 'Pinch with thumb and index'},
        {'name': 'C_SHAPE', 'hamnosys': 'hamceeall', 'description': 'C shape with all fingers'},
        {'name': 'HOOK', 'hamnosys': 'hamfingerhookmod', 'description': 'Hooked fingers'},
        {'name': 'BENT', 'hamnosys': 'hamfingerbendmod', 'description': 'Bent fingers'},
        {'name': 'THUMB_ONLY', 'hamnosys': 'hamthumb', 'description': 'Only thumb extended'},
        {'name': 'INDEX_ONLY', 'hamnosys': 'hamindexfinger', 'description': 'Only index extended'},
        {'name': 'MIDDLE_ONLY', 'hamnosys': 'hammiddlefinger', 'description': 'Only middle extended'},
        {'name': 'RING_ONLY', 'hamnosys': 'hamringfinger', 'description': 'Only ring extended'},
        {'name': 'PINKY_ONLY', 'hamnosys': 'hampinky', 'description': 'Only pinky extended'},
        {'name': 'DOUBLE_BENT', 'hamnosys': 'hamdoublebent', 'description': 'Double bent fingers'}
    ]
    return jsonify(handshapes)


@app.route('/locations', methods=['GET'])
def get_locations():
    """Return list of supported locations"""
    locations = [
        {'name': 'HEAD_TOP', 'hamnosys': 'hamheadtop', 'description': 'Top of head'},
        {'name': 'FOREHEAD', 'hamnosys': 'hamforehead', 'description': 'Forehead'},
        {'name': 'NOSE', 'hamnosys': 'hamnose', 'description': 'Nose'},
        {'name': 'LIPS', 'hamnosys': 'hamlips', 'description': 'Lips'},
        {'name': 'CHIN', 'hamnosys': 'hamchin', 'description': 'Chin'},
        {'name': 'UNDERCHIN', 'hamnosys': 'hamunderchin', 'description': 'Under chin'},
        {'name': 'NECK', 'hamnosys': 'hamneck', 'description': 'Neck'},
        {'name': 'SHOULDERS', 'hamnosys': 'hamshoulders', 'description': 'Shoulders'},
        {'name': 'CHEST', 'hamnosys': 'hamchest', 'description': 'Chest'},
        {'name': 'STOMACH', 'hamnosys': 'hamstomach', 'description': 'Stomach'},
        {'name': 'EYES', 'hamnosys': 'hameyes', 'description': 'Eyes'},
        {'name': 'EARS', 'hamnosys': 'hamear', 'description': 'Ears'},
        {'name': 'CHEEK', 'hamnosys': 'hamcheek', 'description': 'Cheek'},
        {'name': 'PALM', 'hamnosys': 'hampalm', 'description': 'Palm of hand'},
        {'name': 'HANDBACK', 'hamnosys': 'hamhandback', 'description': 'Back of hand'}
    ]
    return jsonify(locations)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)