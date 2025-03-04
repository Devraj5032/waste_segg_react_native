import React, { useState, useRef, useEffect } from "react";
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Text,
  Image,
  Button,
  Dimensions,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
  ScrollView,
} from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { MaterialIcons, Ionicons, FontAwesome5 } from "@expo/vector-icons";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as jpeg from "jpeg-js";
import { decodeJpeg } from "@tensorflow/tfjs-react-native";
import { LinearGradient } from "expo-linear-gradient";
import * as ImageManipulator from "expo-image-manipulator";

const { width, height } = Dimensions.get("window");

// Model input size - 640x640 square
const MODEL_SIZE = 640;

export default function QRScannerInput() {
  const [scannedData, setScannedData] = useState("");
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isImageCameraOpen, setIsImageCameraOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [capturedImages, setCapturedImages] = useState([null, null, null]); // Array for 3 images
  const [imagePredictions, setImagePredictions] = useState([null, null, null]); // Individual predictions
  const [currentImageIndex, setCurrentImageIndex] = useState(0); // Track which image is being captured
  const [isModelReady, setIsModelReady] = useState(false);
  const [finalPrediction, setFinalPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processingImageIndex, setProcessingImageIndex] = useState(null); // Track which image is being processed
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const model = useRef(null);

  // Calculate square crop size for camera view
  const cropSize = Math.min(width, height) * 0.8; // 80% of the smaller dimension

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setLoading(true);
      await tf.ready();
      
      const modelJson = require("../assets/model/model.json");
      const modelWeights = [
        require("../assets/model/group1-shard1of3.bin"),
        require("../assets/model/group1-shard2of3.bin"),
        require("../assets/model/group1-shard3of3.bin"),
      ];
  
      model.current = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setIsModelReady(true);
      setLoading(false);
      console.log("Model loaded successfully");
    } catch (error) {
      console.error("Error loading model:", error);
      setLoading(false);
    }
  };

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionContainer}>
        <LinearGradient
          colors={['#4c669f', '#3b5998', '#192f6a']}
          style={styles.gradientBackground}
        />
        <View style={styles.permissionContent}>
          <Ionicons name="camera-outline" size={80} color="white" />
          <Text style={styles.permissionTitle}>Camera Access Needed</Text>
          <Text style={styles.permissionText}>
            We need camera access to scan QR codes and classify waste images.
          </Text>
          <TouchableOpacity 
            style={styles.permissionButton} 
            onPress={requestPermission}
          >
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  function handleBarcodeScanned({ data }) {
    if (!data.startsWith("HS")) {
      setErrorMessage("Invalid QR Code");
      setScannedData("");
    } else {
      setErrorMessage("");
      setScannedData(data);
    }
    setIsCameraOpen(false);
  }

  async function handleImageCaptured() {
    if (!cameraRef.current) return;
    setLoading(true);
    
    try {
      // Capture the full image
      const photo = await cameraRef.current.takePictureAsync({ quality: 1 });
      
      // Calculate the center crop coordinates
      const photoWidth = photo.width;
      const photoHeight = photo.height;
      const size = Math.min(photoWidth, photoHeight);
      const originX = (photoWidth - size) / 2;
      const originY = (photoHeight - size) / 2;
      
      // Crop to square and resize to MODEL_SIZE
      const manipResult = await ImageManipulator.manipulateAsync(
        photo.uri,
        [
          { 
            crop: { 
              originX, 
              originY, 
              width: size, 
              height: size 
            } 
          },
          { resize: { width: MODEL_SIZE, height: MODEL_SIZE } }
        ],
        { compress: 1, format: ImageManipulator.SaveFormat.JPEG }
      );
      
      // Update the current image in the array
      const newImages = [...capturedImages];
      newImages[currentImageIndex] = manipResult.uri;
      setCapturedImages(newImages);
      
      // Process this image immediately
      setProcessingImageIndex(currentImageIndex);
      const prediction = await runModelOnImage(manipResult.uri);
      
      // Update the prediction for this image
      const newPredictions = [...imagePredictions];
      newPredictions[currentImageIndex] = prediction;
      setImagePredictions(newPredictions);
      
      // If we've captured all 3 images, close the camera and calculate final prediction
      if (currentImageIndex >= 2) {
        setIsImageCameraOpen(false);
        calculateFinalPrediction([...newPredictions]);
        setCurrentImageIndex(0); // Reset for next time
      } else {
        // Otherwise, increment to the next image
        setCurrentImageIndex(currentImageIndex + 1);
      }
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setLoading(false);
      setProcessingImageIndex(null);
    }
  }

  async function runModelOnImage(imageUri) {
    if (!model.current) return null;
    try {
      const response = await fetch(imageUri);
      const imageData = await response.blob();
  
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
        reader.onloadend = async () => {
          try {
            const buffer = new Uint8Array(reader.result);
            let imageTensor = decodeJpeg(buffer);
        
            // Image is already resized to MODEL_SIZE x MODEL_SIZE, just normalize
            const processedImage = imageTensor
              .toFloat()
              .div(tf.scalar(255))
              .expandDims();
        
            console.log("Running model with resized image...");
        
            const predictions = model.current.predict(processedImage);
            const predictionData = await predictions.data();
        
            // Get the index of the highest probability
            const predictedIndex = predictionData.indexOf(Math.max(...predictionData));
        
            // Class labels
            const classNames = ["dry waste", "wet waste"];
            const predictedClass = classNames[predictedIndex] || "Unknown";
            
            // Clean up tensors
            tf.dispose([imageTensor, processedImage, predictions]);
            
            resolve(predictedClass);
          } catch (error) {
            reject(error);
          }
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(imageData);
      });
    } catch (error) {
      console.error("Error running model:", error);
      return null;
    }
  }

  function calculateFinalPrediction(predictions) {
    // Count occurrences of each class
    const counts = predictions.reduce((acc, pred) => {
      if (pred) {
        acc[pred] = (acc[pred] || 0) + 1;
      }
      return acc;
    }, {});
    
    // Find the most common prediction (majority vote)
    let finalResult = null;
    let maxCount = 0;
    
    for (const [pred, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        finalResult = pred;
      }
    }
    
    console.log(`Final prediction (${maxCount}/3 votes): ${finalResult}`);
    setFinalPrediction(finalResult);
    return finalResult;
  }

  function resetImages() {
    setCapturedImages([null, null, null]);
    setImagePredictions([null, null, null]);
    setCurrentImageIndex(0);
    setFinalPrediction(null);
  }

  // Get badge color based on prediction
  const getPredictionColor = (prediction) => {
    if (!prediction) return "#999";
    return prediction === "dry waste" ? "#4BB543" : "#3b5998";
  };

  // Get icon based on prediction
  const getPredictionIcon = (prediction) => {
    if (!prediction) return null;
    return prediction === "dry waste" ? "trash-alt" : "water";
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />
      
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Waste Classifier</Text>
        {loading && <ActivityIndicator size="small" color="#4c669f" />}
      </View>
      
      <View style={styles.modelStatus}>
        <View style={[styles.statusIndicator, { backgroundColor: isModelReady ? '#4BB543' : '#FFD700' }]} />
        <Text style={styles.statusText}>
          {isModelReady ? "AI Model Ready" : "Loading AI Model..."}
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Scan QR Code</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={scannedData}
              placeholder="QR Code will appear here"
              editable={false}
              placeholderTextColor="#9e9e9e"
            />
            <TouchableOpacity 
              style={styles.scanButton} 
              onPress={() => setIsCameraOpen(true)}
            >
              <MaterialIcons name="qr-code-scanner" size={24} color="white" />
            </TouchableOpacity>
          </View>
          {errorMessage ? (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle" size={18} color="#ff3b30" />
              <Text style={styles.errorText}>{errorMessage}</Text>
            </View>
          ) : null}
        </View>

        {scannedData && !errorMessage && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Waste Classification</Text>
            
            <View style={styles.imageCountStatus}>
              <Text style={styles.imageCountText}>
                Images: {capturedImages.filter(img => img !== null).length}/3
              </Text>
              <View style={styles.imageDots}>
                {capturedImages.map((img, index) => (
                  <View 
                    key={index} 
                    style={[
                      styles.imageDot, 
                      { 
                        backgroundColor: img ? '#4c669f' : '#e0e0e0',
                        borderColor: index === currentImageIndex && isImageCameraOpen ? '#ff3b30' : 'transparent'
                      }
                    ]} 
                  />
                ))}
              </View>
            </View>
            
            <View style={styles.imagesContainer}>
              {capturedImages.map((image, index) => (
                <View key={index} style={styles.imageColumn}>
                  <View style={styles.imageWrapper}>
                    {image ? (
                      <>
                        <Image source={{ uri: image }} style={styles.previewImage} />
                        {processingImageIndex === index && (
                          <View style={styles.processingOverlay}>
                            <ActivityIndicator color="white" size="small" />
                          </View>
                        )}
                      </>
                    ) : (
                      <View style={styles.emptyImage}>
                        <Text style={styles.emptyImageText}>{index + 1}</Text>
                      </View>
                    )}
                  </View>
                  
                  {/* Individual prediction badge */}
                  {imagePredictions[index] && (
                    <View style={[
                      styles.individualPredictionBadge, 
                      { backgroundColor: getPredictionColor(imagePredictions[index]) }
                    ]}>
                      {getPredictionIcon(imagePredictions[index]) && (
                        <FontAwesome5 
                          name={getPredictionIcon(imagePredictions[index])} 
                          size={12} 
                          color="white" 
                        />
                      )}
                      <Text style={styles.individualPredictionText}>
                        {imagePredictions[index]?.split(' ')[0]}
                      </Text>
                    </View>
                  )}
                </View>
              ))}
            </View>
            
            {/* Final prediction section */}
            {capturedImages.some(img => img !== null) && (
              <View style={styles.finalPredictionContainer}>
                <Text style={styles.finalPredictionLabel}>Final Classification:</Text>
                {finalPrediction ? (
                  <View style={[
                    styles.predictionBadge, 
                    { backgroundColor: getPredictionColor(finalPrediction) }
                  ]}>
                    <FontAwesome5 
                      name={getPredictionIcon(finalPrediction)} 
                      size={20} 
                      color="white" 
                    />
                    <Text style={styles.predictionText}>
                      {finalPrediction.toUpperCase()}
                    </Text>
                  </View>
                ) : (
                  <Text style={styles.waitingText}>
                    {capturedImages.every(img => img !== null) 
                      ? "Calculating final result..." 
                      : "Capture all 3 images for final result"}
                  </Text>
                )}
              </View>
            )}
            
            <View style={styles.buttonRow}>
              <TouchableOpacity 
                style={[styles.captureImageButton, { flex: 1 }]} 
                onPress={() => {
                  setIsImageCameraOpen(true);
                }}
              >
                <Ionicons name="camera" size={20} color="white" />
                <Text style={styles.captureImageButtonText}>
                  {capturedImages.some(img => img !== null) ? "Continue Capturing" : "Take Photos"}
                </Text>
              </TouchableOpacity>
              
              {capturedImages.some(img => img !== null) && (
                <TouchableOpacity 
                  style={[styles.resetButton, { flex: 0.4 }]} 
                  onPress={resetImages}
                >
                  <Ionicons name="refresh" size={20} color="white" />
                </TouchableOpacity>
              )}
            </View>
          </View>
        )}
      </ScrollView>

      {isCameraOpen && (
        <View style={styles.cameraContainer}>
          <CameraView
            style={styles.fullscreenCamera}
            facing="back"
            barcodeScannerEnabled
            onBarcodeScanned={handleBarcodeScanned}
          >
            <View style={styles.scannerOverlay}>
              <View style={styles.scannerFrame} />
            </View>
            <TouchableOpacity 
              style={styles.closeButton} 
              onPress={() => setIsCameraOpen(false)}
            >
              <Ionicons name="close" size={24} color="white" />
            </TouchableOpacity>
            <Text style={styles.scanInstructionText}>
              Position QR code within the frame
            </Text>
          </CameraView>
        </View>
      )}

      {isImageCameraOpen && (
        <View style={styles.cameraContainer}>
          <CameraView 
            ref={cameraRef} 
            style={styles.fullscreenCamera} 
            facing="back"
          >
            <View style={styles.captureOverlay}>
              {/* Square crop guide matching model input size */}
              <View style={[styles.cropGuide, { width: cropSize, height: cropSize }]}>
                <View style={styles.cropCorner} />
                <View style={[styles.cropCorner, { right: 0 }]} />
                <View style={[styles.cropCorner, { bottom: 0 }]} />
                <View style={[styles.cropCorner, { right: 0, bottom: 0 }]} />
                <Text style={styles.cropSizeText}>{MODEL_SIZE}×{MODEL_SIZE}</Text>
              </View>
            </View>
            
            <TouchableOpacity 
              style={styles.closeButton} 
              onPress={() => setIsImageCameraOpen(false)}
            >
              <Ionicons name="close" size={24} color="white" />
            </TouchableOpacity>
            
            <View style={styles.captureControls}>
              <TouchableOpacity 
                style={styles.captureButton} 
                onPress={handleImageCaptured}
                disabled={loading}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>
            </View>
            
            <Text style={styles.captureInstructionText}>
              Image {currentImageIndex + 1}/3: Center waste item in square
            </Text>
          </CameraView>
        </View>
      )}
      
      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="white" />
          <Text style={styles.loadingText}>Processing image...</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f9fa",
  },
  scrollContent: {
    paddingBottom: 20,
  },
  gradientBackground: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  permissionContent: {
    alignItems: "center",
    padding: 20,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
    marginTop: 20,
    marginBottom: 10,
  },
  permissionText: {
    fontSize: 16,
    color: "white",
    textAlign: "center",
    marginBottom: 30,
  },
  permissionButton: {
    backgroundColor: "white",
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 25,
    elevation: 3,
  },
  permissionButtonText: {
    color: "#3b5998",
    fontSize: 16,
    fontWeight: "bold",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#333",
  },
  modelStatus: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: "#f0f0f0",
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusText: {
    fontSize: 14,
    color: "#555",
  },
  card: {
    margin: 16,
    padding: 16,
    backgroundColor: "white",
    borderRadius: 12,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 12,
    color: "#333",
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e0e0e0",
    borderRadius: 8,
    backgroundColor: "#f9f9f9",
    overflow: "hidden",
  },
  input: {
    flex: 1,
    height: 50,
    paddingHorizontal: 12,
    fontSize: 16,
    color: "#333",
  },
  scanButton: {
    backgroundColor: "#4c669f",
    height: 50,
    width: 50,
    justifyContent: "center",
    alignItems: "center",
  },
  errorContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 8,
  },
  errorText: {
    color: "#ff3b30",
    marginLeft: 6,
    fontSize: 14,
  },
  resultCard: {
    margin: 16,
    padding: 16,
    backgroundColor: "white",
    borderRadius: 12,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 16,
    color: "#333",
  },
  imageCountStatus: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  imageCountText: {
    fontSize: 14,
    color: "#666",
  },
  imageDots: {
    flexDirection: "row",
  },
  imageDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginLeft: 5,
    borderWidth: 2,
  },
  imagesContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  imageColumn: {
    alignItems: "center",
    width: width * 0.25,
  },
  imageWrapper: {
    width: width * 0.25,
    height: width * 0.25,
    borderRadius: 8,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#e0e0e0",
    position: "relative",
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  processingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  emptyImage: {
    width: "100%",
    height: "100%",
    backgroundColor: "#f0f0f0",
    justifyContent: "center",
    alignItems: "center",
  },
  emptyImageText: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#999",
  },
  individualPredictionBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 12,
    marginTop: 6,
  },
  individualPredictionText: {
    color: "white",
    fontWeight: "bold",
    fontSize: 12,
    marginLeft: 4,
  },
  finalPredictionContainer: {
    alignItems: "center",
    marginVertical: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: "#e0e0e0",
  },
  finalPredictionLabel: {
    fontSize: 16,
    fontWeight: "600",
    color: "#333",
    marginBottom: 8,
  },
  predictionBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  predictionText: {
    color: "white",
    fontWeight: "bold",
    fontSize: 16,
    marginLeft: 8,
  },
  waitingText: {
    fontSize: 16,
    color: "#666",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
  captureImageButton: {
    flexDirection: "row",
    backgroundColor: "#4c669f",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
  },
  captureImageButtonText: {
    color: "white",
    fontWeight: "600",
    fontSize: 16,
    marginLeft: 8,
  },
  resetButton: {
    backgroundColor: "#ff3b30",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    alignItems: "center",
    justifyContent: "center",
  },
  cameraContainer: {
    position: "absolute",
    width: width,
    height: height,
    top: 0,
    left: 0,
    zIndex: 100,
  },
  fullscreenCamera: {
    flex: 1,
  },
  scannerOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  scannerFrame: {
    width: 250,
    height: 250,
    borderWidth: 2,
    borderColor: "white",
    borderRadius: 12,
  },
  captureOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  cropGuide: {
    borderWidth: 2,
    borderColor: "white",
    position: "relative",
  },
  cropCorner: {
    position: "absolute",
    width: 20,
    height: 20,
    borderColor: "white",
    borderTopWidth: 4,
    borderLeftWidth: 4,
  },
  cropSizeText: {
    position: "absolute",
    top: -30,
    alignSelf: "center",
    color: "white",
    fontSize: 14,
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
  },
  closeButton: {
    position: "absolute",
    top: 50,
    right: 20,
    backgroundColor: "rgba(0,0,0,0.6)",
    padding: 10,
    borderRadius: 30,
    width: 50,
    height: 50,
    justifyContent: "center",
    alignItems: "center",
  },
  scanInstructionText: {
    position: "absolute",
    bottom: 100,
    alignSelf: "center",
    color: "white",
    fontSize: 16,
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  captureControls: {
    position: "absolute",
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: "rgba(255,255,255,0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "white",
  },
  captureInstructionText: {
    position: "absolute",
    top: 50,
    alignSelf: "center",
    color: "white",
    fontSize: 16,
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 200,
  },
  loadingText: {
    color: "white",
    marginTop: 10,
    fontSize: 16,
  },
});