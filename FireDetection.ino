#include "mlp_fire_model.h"
#include "mobilenetv2_fire_quant.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_camera.h"
#include "camera_pins.h"
#include <DHT.h>

// Pin definitions
#define DHTPIN 2
#define DHTTYPE DHT22
#define MQ2_PIN 4
#define FLAME_PIN 15
#define BUZZER_PIN 13

DHT dht(DHTPIN, DHTTYPE);

// Scaler values (MLP: LPG, CO, smoke, temp, humidity)
float scaler_mean[5] = {0.3436076, 2.25383826, 2.55535601, 37.55112147, 23.42751992};
float scaler_std[5] = {0.35424789, 2.32093111, 2.53485751, 7.24319686, 14.1390869};

// Tensor arenas (PSRAM enabled)
uint8_t mlp_tensor_arena[16 * 1024];
uint8_t mobilenet_tensor_arena[300 * 1024];

// Interpreters
tflite::AllOpsResolver mlp_resolver;
tflite::AllOpsResolver mobilenet_resolver;
tflite::MicroInterpreter mlp_static_interpreter(
    tflite::GetModel(mlp_fire_model), mlp_resolver, mlp_tensor_arena, 16 * 1024);
tflite::MicroInterpreter* mobilenet_interpreter = nullptr;

const float FIRE_THRESHOLD = 0.7;
const float MOBILENET_THRESHOLD = 0.5;

// Sensor reading functions
float readMQ2_LPG() { return analogRead(MQ2_PIN) * 0.1; }
float readMQ2_CO() { return analogRead(MQ2_PIN) * 0.1; }
float readMQ2_Smoke() { return analogRead(MQ2_PIN) * 0.1; }
bool readFlame() { return digitalRead(FLAME_PIN) == LOW; }

void scale_inputs(float* raw_inputs, float* scaled_inputs) {
  for (int i = 0; i < 5; i++) {
    scaled_inputs[i] = (raw_inputs[i] - scaler_mean[i]) / scaler_std[i];
  }
}

void init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (1);
  }
}

float run_mobilenetv2() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return -1.0;
  }

  uint8_t* rgb888 = (uint8_t*)malloc(224 * 224 * 3);
  if (!rgb888) {
    esp_camera_fb_return(fb);
    return -1.0;
  }

  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      int src_x = x * fb->width / 224;
      int src_y = y * fb->height / 224;
      uint16_t pixel = ((uint16_t*)fb->buf)[src_y * fb->width + src_x];
      rgb888[(y * 224 + x) * 3 + 0] = (pixel >> 11) << 3;
      rgb888[(y * 224 + x) * 3 + 1] = ((pixel >> 5) & 0x3F) << 2;
      rgb888[(y * 224 + x) * 3 + 2] = (pixel & 0x1F) << 3;
    }
  }

  float* input_data = (float*)malloc(224 * 224 * 3 * sizeof(float));
  for (int i = 0; i < 224 * 224 * 3; i++) {
    input_data[i] = rgb888[i] / 255.0;
  }

  if (!mobilenet_interpreter) {
    mobilenet_interpreter = new tflite::MicroInterpreter(
        tflite::GetModel(mobilenetv2_fire_quant), mobilenet_resolver, mobilenet_tensor_arena, 300 * 1024);
    if (mobilenet_interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("MobileNetV2 allocation failed!");
      delete mobilenet_interpreter;
      mobilenet_interpreter = nullptr;
    }
  }

  TfLiteTensor* input = mobilenet_interpreter->input(0);
  memcpy(input->data.f, input_data, 224 * 224 * 3 * sizeof(float));
  if (mobilenet_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("MobileNetV2 inference failed!");
    free(rgb888);
    free(input_data);
    esp_camera_fb_return(fb);
    return -1.0;
  }

  TfLiteTensor* output = mobilenet_interpreter->output(0);
  float fire_prob = output->data.f[0];

  free(rgb888);
  free(input_data);
  esp_camera_fb_return(fb);
  return fire_prob;
}

void sound_buzzer() {
  tone(BUZZER_PIN, 1000);
  delay(1000);
  noTone(BUZZER_PIN);
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(FLAME_PIN, INPUT);
  pinMode(MQ2_PIN, INPUT);
  digitalWrite(4, LOW);  // Disable flash LED

  if (mlp_static_interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("MLP allocation failed!");
    while (1);
  }

  init_camera();
  Serial.print("PSRAM Size: ");
  Serial.println(ESP.getPsramSize());
  Serial.println("Setup complete");
}

void loop() {
  float raw_inputs[5];
  raw_inputs[0] = readMQ2_LPG();
  raw_inputs[1] = readMQ2_CO();
  raw_inputs[2] = readMQ2_Smoke();
  raw_inputs[3] = dht.readTemperature();
  raw_inputs[4] = dht.readHumidity();

  bool flame_detected = readFlame();

  if (isnan(raw_inputs[3]) || isnan(raw_inputs[4])) {
    Serial.println("DHT22 reading failed!");
    delay(2000);
    return;
  }

  float scaled_inputs[5];
  scale_inputs(raw_inputs, scaled_inputs);

  Serial.print("Raw Inputs: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(raw_inputs[i]);
    Serial.print(" ");
  }
  Serial.print("Flame Detected: ");
  Serial.println(flame_detected ? "Yes" : "No");

  TfLiteTensor* mlp_input = mlp_static_interpreter.input(0);
  for (int i = 0; i < 5; i++) {
    mlp_input->data.f[i] = scaled_inputs[i];
  }
  if (mlp_static_interpreter.Invoke() != kTfLiteOk) {
    Serial.println("MLP inference failed!");
    return;
  }

  TfLiteTensor* mlp_output = mlp_static_interpreter.output(0);
  float mlp_fire_prob = mlp_output->data.f[0];
  Serial.print("MLP Fire Probability: ");
  Serial.println(mlp_fire_prob);

  if (mlp_fire_prob >= FIRE_THRESHOLD || flame_detected) {
    Serial.println("Triggering MobileNetV2...");
    float mobilenet_fire_prob = run_mobilenetv2();
    if (mobilenet_fire_prob >= 0) {
      Serial.print("MobileNetV2 Fire Probability: ");
      Serial.println(mobilenet_fire_prob);
      if (mobilenet_fire_prob > MOBILENET_THRESHOLD) {
        Serial.println("Fire confirmed!");
        sound_buzzer();
      } else {
        Serial.println("No fire confirmed by MobileNetV2.");
      }
    }
  } else {
    Serial.println("No fire detected by MLP or flame sensor.");
  }

  delay(2000);
}