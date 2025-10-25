#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <vector>
#include <algorithm>    // std::sort() için
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp> // NMS için
#include <QPixmap>
#include <QLabel>
#include <QResizeEvent>
#include <QIcon>



float MainWindow::calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2)
{
    cv::Rect intersection = rect1 & rect2;
    float intersectionArea = (float)intersection.area();
    float unionArea = (float)(rect1.area() + rect2.area() - intersectionArea);
    if (unionArea < 1e-5) return 0.0f;
    return intersectionArea / unionArea;
}

void MainWindow::performNMS(std::vector<FaceProposal>& proposals, float iou_threshold, std::vector<FaceProposal>& final_faces)
{
    if (proposals.empty()) return;
    std::sort(proposals.begin(), proposals.end(), [](const FaceProposal& a, const FaceProposal& b) {
        return a.score > b.score;
    });
    std::vector<bool> suppressed(proposals.size(), false);
    for (size_t i = 0; i < proposals.size(); ++i)
    {
        if (suppressed[i]) continue;
        final_faces.push_back(proposals[i]);
        for (size_t j = i + 1; j < proposals.size(); ++j)
        {
            if (suppressed[j]) continue;
            float iou = calculateIOU(proposals[i].rect, proposals[j].rect);
            if (iou > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }
}

QImage MainWindow::cvMatToQImage(const cv::Mat &mat)
{
    if(mat.type() == CV_8UC3)
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped();
    else if(mat.type() == CV_8UC1)
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    else
        return QImage();
}


std::vector<FaceProposal> MainWindow::detectFacesInImage(const cv::Mat& image)
{
    std::vector<FaceProposal> final_faces;

    try {
        const int det_input_size = 640;
        const float original_width = (float)image.cols;
        const float original_height = (float)image.rows;

        // Görüntüyü yeniden boyutlandır (orantıyı koruyarak)
        cv::Mat resized;
        float scale = std::min((float)det_input_size / original_width,
                               (float)det_input_size / original_height);
        int new_width = (int)(original_width * scale);
        int new_height = (int)(original_height * scale);

        cv::resize(image, resized, cv::Size(new_width, new_height));

        // Kenarları siyah ile doldurarak 640x640 yap
        cv::Mat padded = cv::Mat::zeros(det_input_size, det_input_size, CV_8UC3);
        int x_offset = (det_input_size - new_width) / 2;
        int y_offset = (det_input_size - new_height) / 2;
        resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));

        // Blob hazırlama
        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0/128.0,
                               cv::Size(det_input_size, det_input_size),
                               cv::Scalar(127.5, 127.5, 127.5),
                               true, false, CV_32F);

        std::vector<int64_t> inputShape = {1, 3, det_input_size, det_input_size};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                                 blob.ptr<float>(),
                                                                 blob.total(),
                                                                 inputShape.data(),
                                                                 inputShape.size());

        // Model çalıştır
        const char* inputNames[] = {"input.1"};
        const char* outputNames[] = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};

        auto outputTensors = detector->Run(Ort::RunOptions{nullptr},
                                           inputNames, &inputTensor, 1,
                                           outputNames, 9);

        // Post-processing
        std::vector<FaceProposal> candidates;
        const float score_threshold = 0.5f;
        const std::vector<int> strides = {8, 16, 32};
        const std::vector<int> num_anchors_list = {12800, 3200, 800};
        const std::vector<int> feat_sizes = {80, 40, 20};

        for (int i = 0; i < strides.size(); i++)
        {
            int stride = strides[i];
            int feat_size = feat_sizes[i];
            int num_anchors = num_anchors_list[i];

            float* scores = outputTensors[i + 0].GetTensorMutableData<float>();
            float* bboxes = outputTensors[i + 3].GetTensorMutableData<float>();
            float* landmarks = outputTensors[i + 6].GetTensorMutableData<float>();

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                float score = scores[anchor_idx];
                if (score < score_threshold) continue;

                int grid_index = anchor_idx / 2;
                int grid_y = grid_index / feat_size;
                int grid_x = grid_index % feat_size;

                float center_x = (grid_x + 0.5f) * stride;
                float center_y = (grid_y + 0.5f) * stride;

                float dist_left = bboxes[anchor_idx * 4 + 0];
                float dist_top = bboxes[anchor_idx * 4 + 1];
                float dist_right = bboxes[anchor_idx * 4 + 2];
                float dist_bottom = bboxes[anchor_idx * 4 + 3];

                float x1 = center_x - dist_left * stride;
                float y1 = center_y - dist_top * stride;
                float x2 = center_x + dist_right * stride;
                float y2 = center_y + dist_bottom * stride;

                // Koordinat dönüşümü
                float x1_unpad = (x1 - x_offset) / scale;
                float y1_unpad = (y1 - y_offset) / scale;
                float x2_unpad = (x2 - x_offset) / scale;
                float y2_unpad = (y2 - y_offset) / scale;

                if (x2_unpad <= x1_unpad || y2_unpad <= y1_unpad) {
                    continue;
                }

                x1_unpad = std::max(0.0f, std::min(x1_unpad, original_width - 1));
                y1_unpad = std::max(0.0f, std::min(y1_unpad, original_height - 1));
                x2_unpad = std::max(1.0f, std::min(x2_unpad, original_width));
                y2_unpad = std::max(1.0f, std::min(y2_unpad, original_height));

                // Landmarks
                std::vector<cv::Point2f> lms;
                for(int k = 0; k < 5; k++) {
                    float lx = center_x + landmarks[anchor_idx * 10 + k * 2 + 0] * stride;
                    float ly = center_y + landmarks[anchor_idx * 10 + k * 2 + 1] * stride;

                    float lx_unpad = (lx - x_offset) / scale;
                    float ly_unpad = (ly - y_offset) / scale;

                    lx_unpad = std::max(0.0f, std::min(lx_unpad, original_width - 1));
                    ly_unpad = std::max(0.0f, std::min(ly_unpad, original_height - 1));

                    lms.push_back(cv::Point2f(lx_unpad, ly_unpad));
                }

                FaceProposal proposal;
                proposal.score = score;
                proposal.rect = cv::Rect(
                    cv::Point((int)x1_unpad, (int)y1_unpad),
                    cv::Point((int)x2_unpad, (int)y2_unpad)
                    );

                if (proposal.rect.width < 20 || proposal.rect.height < 20) {
                    continue;
                }

                proposal.landmarks = lms;
                candidates.push_back(proposal);
            }
        }

        // NMS uygula
        performNMS(candidates, 0.3f, final_faces);

        // Yüksek güvenilirlikteki yüzleri filtrele
        std::vector<FaceProposal> high_confidence_faces;
        for (const auto& face : final_faces) {
            if (face.score >= 0.6f) {
                high_confidence_faces.push_back(face);
            }
        }
        final_faces = high_confidence_faces;

    } catch (const std::exception& e) {
        qDebug() << "Face detection error:" << e.what();
    }

    return final_faces;
}
std::vector<float> MainWindow::getFaceEmbedding(const cv::Mat& image, const cv::Rect& faceRect)
{
    std::vector<float> embedding;

    try {
        // Yüzü kırp ve 112x112 boyutuna getir
        cv::Mat faceCrop = image(faceRect).clone();

        // Debug: Kırpılan yüzün boyutunu kontrol et
        if(faceCrop.empty()) {
            qDebug() << "Hata: Kırpılan yüz boş!";
            return embedding;
        }

        cv::resize(faceCrop, faceCrop, cv::Size(112, 112));
        cv::cvtColor(faceCrop, faceCrop, cv::COLOR_BGR2RGB);
        faceCrop.convertTo(faceCrop, CV_32FC3, 1.0/127.5, -1.0);

        // Embedding modeli için input hazırlama
        std::vector<int64_t> embedShape = {1, 3, 112, 112};
        std::vector<float> embedValues(3 * 112 * 112);

        // CHW formatına dönüştür
        for(int c = 0; c < 3; c++) {
            for(int row = 0; row < 112; row++) {
                for(int col = 0; col < 112; col++) {
                    embedValues[c * 112 * 112 + row * 112 + col] = faceCrop.at<cv::Vec3f>(row, col)[c];
                }
            }
        }

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value embedInput = Ort::Value::CreateTensor<float>(memoryInfo,
                                                                embedValues.data(),
                                                                embedValues.size(),
                                                                embedShape.data(),
                                                                embedShape.size());

        // DÜZELTME: Output ismini "fc1" yerine "683" yap
        const char* embedInputNames[] = {"input.1"};
        const char* embedOutputNames[] = {"683"};  // <-- BURASI DÜZELDİ

        auto embedOutput = embedder->Run(Ort::RunOptions{nullptr},
                                         embedInputNames, &embedInput, 1,
                                         embedOutputNames, 1);

        float* embedding_data = embedOutput.front().GetTensorMutableData<float>();
        size_t embedding_size = embedOutput.front().GetTensorTypeAndShapeInfo().GetElementCount();

        embedding = std::vector<float>(embedding_data, embedding_data + embedding_size);

        qDebug() << "Embedding boyutu:" << embedding.size(); // 512 olmalı

    } catch (const std::exception& e) {
        qDebug() << "Embedding extraction error:" << e.what();
    }

    return embedding;
}
float MainWindow::cosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2)
{
    if (vec1.size() != vec2.size() || vec1.empty()) {
        return 0.0f;
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); i++) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    if (norm1 < 1e-10 || norm2 < 1e-10) {
        return 0.0f;
    }

    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

void MainWindow::drawFaceDetectionResult(cv::Mat& image, const std::vector<FaceProposal>& faces)
{
    for (size_t i = 0; i < faces.size(); i++) {
        const FaceProposal& face = faces[i];

        // Bounding box çiz
        cv::rectangle(image, face.rect, cv::Scalar(0, 255, 0), 2);

        // Landmark'ları çiz
        for (const auto& lm : face.landmarks) {
            cv::circle(image, lm, 3, cv::Scalar(0, 0, 255), -1);
        }

        // Skor bilgisini yaz
        std::string score_text = "Score: " + std::to_string(face.score).substr(0, 4);
        cv::putText(image, score_text,
                    cv::Point(face.rect.x, face.rect.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // Yüz numarasını yaz
        std::string face_text = "Face " + std::to_string(i + 1);
        cv::putText(image, face_text,
                    cv::Point(face.rect.x, face.rect.y - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    // Member'daki pixmap'leri kullanarak labelleri yeniden ölçeklendir
    updateTab1Pixmap();
    updateTab2SourcePixmap();
    updateTab2TargetPixmap();
}
// Bu 3 fonksiyonu mainwindow.cpp dosyanıza (örn: resizeEvent'in üstüne) ekleyin

void MainWindow::updateTab1Pixmap()
{
    // Eğer pixmap boşsa (örn: program yeni başladıysa) default olanı yükle
    if (m_pixmapTab1.isNull()) {
        m_pixmapTab1 = QPixmap(":/images/resources/face_placeholder.png");
    }

    // Label boyutunu al ve pixmap'i ölçeklendir
    int width = ui->label_image_tab_1->width() > 10 ? ui->label_image_tab_1->width() : 100;
    int height = ui->label_image_tab_1->height() > 10 ? ui->label_image_tab_1->height() : 100;

    ui->label_image_tab_1->setPixmap(
        m_pixmapTab1.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );
}

void MainWindow::updateTab2SourcePixmap()
{
    if (m_pixmapTab2Source.isNull()) {
        m_pixmapTab2Source = QPixmap(":/images/resources/face_placeholder.png");
    }
    int width = ui->label_source_face_tab_2->width() > 10 ? ui->label_source_face_tab_2->width() : 100;
    int height = ui->label_source_face_tab_2->height() > 10 ? ui->label_source_face_tab_2->height() : 100;

    ui->label_source_face_tab_2->setPixmap(
        m_pixmapTab2Source.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );
}

void MainWindow::updateTab2TargetPixmap()
{
    if (m_pixmapTab2Target.isNull()) {
        m_pixmapTab2Target = QPixmap(":/images/resources/face_placeholder.png");
    }
    int width = ui->label_target_face_tab_2->width() > 10 ? ui->label_target_face_tab_2->width() : 100;
    int height = ui->label_target_face_tab_2->height() > 10 ? ui->label_target_face_tab_2->height() : 100;

    ui->label_target_face_tab_2->setPixmap(
        m_pixmapTab2Target.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , env(ORT_LOGGING_LEVEL_WARNING, "FaceRecognition")
    , detector(nullptr)
    , embedder(nullptr)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/resources/app_icon.png"));

    // UI başlangıç ayarları
    updateTab1Pixmap();
    updateTab2SourcePixmap();
    updateTab2TargetPixmap();

    ui->label_image_tab_1->setAlignment(Qt::AlignCenter);
    ui->label_source_face_tab_2->setAlignment(Qt::AlignCenter);
    ui->label_target_face_tab_2->setAlignment(Qt::AlignCenter);

    setAboutText();
    this->setWindowTitle("FaceComparator | CXX Edition");
    ui->progressBar_sim_rate_tab_2->setValue(0);

    try
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Executable'ın bulunduğu dizinden model yollarını oluştur
        QDir appDir(QCoreApplication::applicationDirPath());
        QString modelsPath = appDir.filePath("models");
        QString detModelPath = QDir(modelsPath).filePath("det_10g.onnx");
        QString embModelPath = QDir(modelsPath).filePath("w600k_r50.onnx");

        // Debug: Model yollarını kontrol et
        qDebug() << "Models directory:" << modelsPath;
        qDebug() << "Detection model path:" << detModelPath;
        qDebug() << "Embedding model path:" << embModelPath;

        // Dosyaların var olup olmadığını kontrol et
        if (!QFile::exists(detModelPath)) {
            QMessageBox::critical(this, "Model Hatası",
                                  QString("Tespit modeli bulunamadı:\n%1\n\nLütfen models klasörünü kontrol edin.").arg(detModelPath));
            return;
        }

        if (!QFile::exists(embModelPath)) {
            QMessageBox::critical(this, "Model Hatası",
                                  QString("Embedding modeli bulunamadı:\n%1\n\nLütfen models klasörünü kontrol edin.").arg(embModelPath));
            return;
        }

        // QString'i std::wstring'e çevir
        std::wstring model_detection_path = detModelPath.toStdWString();
        std::wstring model_embedding_path = embModelPath.toStdWString();

        detector = new Ort::Session(env, model_detection_path.c_str(), session_options);
        embedder = new Ort::Session(env, model_embedding_path.c_str(), session_options);

        // Debug için model bilgileri
        Ort::AllocatorWithDefaultOptions allocator;
        auto det_input_name = detector->GetInputNameAllocated(0, allocator);
        auto emb_input_name = embedder->GetInputNameAllocated(0, allocator);
        qDebug() << "Detector model input name:" << det_input_name.get();
        qDebug() << "Embedding model input name:" << emb_input_name.get();

        qDebug() << "Modeller başarıyla yüklendi!";

    }
    catch (const Ort::Exception& e)
    {
        QMessageBox::critical(this, "Model Yükleme Hatası",
                              QString("ONNX modelleri yüklenemedi.\n\nHata: %1").arg(e.what()));
        qApp->quit();
    }
    catch (const std::exception& e)
    {
        QMessageBox::critical(this, "Genel Hata",
                              QString("Beklenmedik bir hata oluştu: %1").arg(e.what()));
        qApp->quit();
    }
}

MainWindow::~MainWindow()
{
    delete ui;
    // Pointer'ları nullptr olarak initialize ettiğimiz için güvenle delete edebiliriz
    if (detector) {
        delete detector;
        detector = nullptr;
    }
    if (embedder) {
        delete embedder;
        embedder = nullptr;
    }
}

void MainWindow::on_pushButton_selectFile_tab_1_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if(filename.isEmpty()) return;

    currentImage = cv::imread(filename.toStdString());
    if(currentImage.empty())
    {
        QMessageBox::warning(this, "Hata", "Görüntü yüklenemedi!");
        return;
    }
    original_image_for_analysis = currentImage.clone();

    ui->label_image_tab_1->setPixmap(QPixmap::fromImage(cvMatToQImage(original_image_for_analysis))
                                         .scaled(ui->label_image_tab_1->size(),
                                                 Qt::KeepAspectRatio,
                                                 Qt::SmoothTransformation));
}
// MainWindow constructor'ında veya ayrı bir fonksiyonda
void MainWindow::setAboutText()
{
    QString aboutText =
        "<html>"
        "<head/>"
        "<body>"
        "<p align='center'><span style='font-size:16pt; font-weight:700; color:#3498db;'>FaceComparator</span></p>"
        "<p align='center'><span style='font-size:12pt; color:#ecf0f1;'>Yüz Tanıma ve Karşılaştırma Sistemi</span></p>"
        "<p align='center'><span style='font-size:10pt; color:#bdc3c7;'>v2.0.1</span></p>"
        "<hr/>"
        "<p><span style='font-weight:600; color:#3498db;'>✨ ÖZELLİKLER:</span></p>"
        "<ul>"
        "<li>Yüksek Hassasiyetli Yüz Tespiti</li>"
        "<li>Detaylı Analiz ve Raporlama</li>"
        "</ul>"
        "<p><span style='font-weight:600; color:#3498db;'>🛠 TEKNOLOJİLER:</span></p>"
        "<ul>"
        "<li>PyTorch &amp; ONNX Runtime</li>"
        "<li>InsightFace Buffalo_L Model</li>"
        "<li>OpenCV Görüntü İşleme</li>"
        "<li>Qt Modern Arayüz</li>"
        "</ul>"
        "<p><span style='font-weight:600; color:#3498db;'>⚖️ LİSANS:</span></p>"
        "<ul>"
        "<li>MIT Lisansı ile lisanslanmıştır</li>"
        "<li>Ticari kullanımı kesinlikle yasaktır</li>"
        "<li>Modellerin kendi lisans koşulları geçerlidir</li>"
        "</ul>"
        "<p><span style='font-weight:600; color:#3498db;'>📞 İLETİŞİM:</span><br/>"
        "Telegram: @EyeOfWebSupport</p>"
        "<p><span style='font-weight:600; color:#3498db;'>👨‍💻 GELİŞTİRİCİ:</span><br/>"
        "Mehmet Yüksel Şekeroğlu tarafından geliştirilmiştir</p>"
        "</body>"
        "</html>";

    ui->label_about->setText(aboutText);
}
void MainWindow::on_pushButton_detect_tab_1_clicked()
{
    try
    {
        if(original_image_for_analysis.empty())
        {
            QMessageBox::warning(this, "Hata", "Görüntü yüklenmedi!");
            return;
        }

        cv::Mat displayImage = original_image_for_analysis.clone();
        const float original_width = (float)original_image_for_analysis.cols;
        const float original_height = (float)original_image_for_analysis.rows;

        // ---- Model Girişi için Görüntüyü Hazırla ----
        const int det_input_size = 640;

        // Görüntüyü yeniden boyutlandır (orantıyı koruyarak)
        cv::Mat resized;
        float scale = std::min((float)det_input_size / original_width,
                               (float)det_input_size / original_height);
        int new_width = (int)(original_width * scale);
        int new_height = (int)(original_height * scale);

        cv::resize(original_image_for_analysis, resized, cv::Size(new_width, new_height));

        // Kenarları siyah ile doldurarak 640x640 yap
        cv::Mat padded = cv::Mat::zeros(det_input_size, det_input_size, CV_8UC3);
        int x_offset = (det_input_size - new_width) / 2;
        int y_offset = (det_input_size - new_height) / 2;
        resized.copyTo(padded(cv::Rect(x_offset, y_offset, new_width, new_height)));

        // ---- BLOB HAZIRLAMA ----
        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0/128.0,
                               cv::Size(det_input_size, det_input_size),
                               cv::Scalar(127.5, 127.5, 127.5),
                               true, false, CV_32F);

        std::vector<int64_t> inputShape = {1, 3, det_input_size, det_input_size};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                                 blob.ptr<float>(),
                                                                 blob.total(),
                                                                 inputShape.data(),
                                                                 inputShape.size());

        // ---- MODEL ÇALIŞTIR ----
        const char* inputNames[] = {"input.1"};
        const char* outputNames[] = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};

        auto outputTensors = detector->Run(Ort::RunOptions{nullptr},
                                           inputNames, &inputTensor, 1,
                                           outputNames, 9);

        // ---- DÜZELTİLMİŞ POST-PROCESSING ----
        std::vector<FaceProposal> candidates;
        const float score_threshold = 0.5f; // Threshold'u yükselttik (daha az false positive)
        const std::vector<int> strides = {8, 16, 32};
        const std::vector<int> num_anchors_list = {12800, 3200, 800};
        const std::vector<int> feat_sizes = {80, 40, 20}; // 640/8, 640/16, 640/32

        for (int i = 0; i < strides.size(); i++)
        {
            int stride = strides[i];
            int feat_size = feat_sizes[i];
            int num_anchors = num_anchors_list[i];

            float* scores = outputTensors[i + 0].GetTensorMutableData<float>();
            float* bboxes = outputTensors[i + 3].GetTensorMutableData<float>();
            float* landmarks = outputTensors[i + 6].GetTensorMutableData<float>();

            for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
            {
                float score = scores[anchor_idx];
                if (score < score_threshold) continue;

                // **DÜZELTME: Doğru grid koordinat hesaplama**
                // Her grid hücresi için 2 anchor var
                int grid_index = anchor_idx / 2;
                int grid_y = grid_index / feat_size;
                int grid_x = grid_index % feat_size;

                // Anchor merkez noktası
                float center_x = (grid_x + 0.5f) * stride;
                float center_y = (grid_y + 0.5f) * stride;

                // **DÜZELTME: Bounding box formatını düzelt**
                // SCRFD formatı: [distance_left, distance_top, distance_right, distance_bottom]
                float dist_left = bboxes[anchor_idx * 4 + 0];
                float dist_top = bboxes[anchor_idx * 4 + 1];
                float dist_right = bboxes[anchor_idx * 4 + 2];
                float dist_bottom = bboxes[anchor_idx * 4 + 3];

                // Bounding box koordinatlarını hesapla
                float x1 = center_x - dist_left * stride;
                float y1 = center_y - dist_top * stride;
                float x2 = center_x + dist_right * stride;
                float y2 = center_y + dist_bottom * stride;

                // **DÜZELTME: Koordinatları orijinal görüntüye dönüştür**
                // 1. Önce padding'i çıkar
                float x1_unpad = x1 - x_offset;
                float y1_unpad = y1 - y_offset;
                float x2_unpad = x2 - x_offset;
                float y2_unpad = y2 - y_offset;

                // 2. Scale'i tersine çevir
                x1_unpad = x1_unpad / scale;
                y1_unpad = y1_unpad / scale;
                x2_unpad = x2_unpad / scale;
                y2_unpad = y2_unpad / scale;

                // **DÜZELTME: Geçerli bounding box kontrolü**
                if (x2_unpad <= x1_unpad || y2_unpad <= y1_unpad) {
                    continue; // Geçersiz bounding box'ı atla
                }

                // Sınırları kontrol et
                x1_unpad = std::max(0.0f, std::min(x1_unpad, original_width - 1));
                y1_unpad = std::max(0.0f, std::min(y1_unpad, original_height - 1));
                x2_unpad = std::max(1.0f, std::min(x2_unpad, original_width)); // en az 1 pixel genişlik
                y2_unpad = std::max(1.0f, std::min(y2_unpad, original_height)); // en az 1 pixel yükseklik

                // Landmarks (5 points)
                std::vector<cv::Point2f> lms;
                for(int k = 0; k < 5; k++) {
                    float lx = center_x + landmarks[anchor_idx * 10 + k * 2 + 0] * stride;
                    float ly = center_y + landmarks[anchor_idx * 10 + k * 2 + 1] * stride;

                    // Landmark'ları da aynı şekilde dönüştür
                    float lx_unpad = (lx - x_offset) / scale;
                    float ly_unpad = (ly - y_offset) / scale;

                    lx_unpad = std::max(0.0f, std::min(lx_unpad, original_width - 1));
                    ly_unpad = std::max(0.0f, std::min(ly_unpad, original_height - 1));

                    lms.push_back(cv::Point2f(lx_unpad, ly_unpad));
                }

                FaceProposal proposal;
                proposal.score = score;
                proposal.rect = cv::Rect(
                    cv::Point((int)x1_unpad, (int)y1_unpad),
                    cv::Point((int)x2_unpad, (int)y2_unpad)
                    );

                // **DÜZELTME: Çok küçük bounding box'ları filtrele**
                if (proposal.rect.width < 20 || proposal.rect.height < 20) {
                    continue; // Çok küçük tespitleri atla
                }

                proposal.landmarks = lms;
                candidates.push_back(proposal);
            }
        }

        qDebug() << "Total candidates before NMS:" << candidates.size();

        // **DÜZELTME: Daha agresif NMS**
        std::vector<FaceProposal> final_faces;
        performNMS(candidates, 0.3f, final_faces); // IOU threshold'u düşürdük

        // **DÜZELTME: Skor threshold'unu sonradan da uygula**
        std::vector<FaceProposal> high_confidence_faces;
        for (const auto& face : final_faces) {
            if (face.score >= 0.6f) { // Yüksek güvenilirlikteki yüzleri al
                high_confidence_faces.push_back(face);
            }
        }

        ui->textBrowser_output_tab_1->append(QString("Bulunan Yüz Sayısı: %1").arg(high_confidence_faces.size()));

        // Sonuçları çiz
        for(size_t i = 0; i < high_confidence_faces.size(); i++)
        {
            FaceProposal& face = high_confidence_faces[i];

            // Debug: Bounding box bilgilerini yazdır
            qDebug() << "Face" << i << "rect:" << face.rect.x << face.rect.y << face.rect.width << "x" << face.rect.height << "score:" << face.score;

            cv::rectangle(displayImage, face.rect, cv::Scalar(0, 255, 0), 2);

            // Landmark'ları çiz
            for(auto& lm : face.landmarks) {
                cv::circle(displayImage, lm, 3, cv::Scalar(0, 0, 255), -1);
            }

            // Skor bilgisini yaz
            cv::putText(displayImage,
                        QString("Score: %1").arg(face.score, 0, 'f', 2).toStdString(),
                        cv::Point(face.rect.x, face.rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            ui->textBrowser_output_tab_1->append(
                QString("Face %1 (Score: %2) [%3,%4 %5x%6]")
                    .arg(i+1).arg(face.score, 0, 'f', 3)
                    .arg(face.rect.x).arg(face.rect.y)
                    .arg(face.rect.width).arg(face.rect.height)
                );
        }

        ui->label_image_tab_1->setPixmap(
            QPixmap::fromImage(cvMatToQImage(displayImage))
                .scaled(ui->label_image_tab_1->size(),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation)
            );

    }
    catch(const Ort::Exception &e)
    {
        QMessageBox::critical(this, "ONNX Hatası",
                              QString("Model çalıştırılırken bir hata oluştu: %1").arg(e.what()));
        qDebug() << "ONNX Runtime Exception:" << e.what();
    }
    catch(const std::exception &e)
    {
        QMessageBox::critical(this, "Genel Hata",
                              QString("Beklenmedik bir hata oluştu: %1").arg(e.what()));
        qDebug() << "Standard Exception:" << e.what();
    }
}

void MainWindow::on_pushButton_select_source_image_tab_2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Kaynak Resmi Seç", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if(filename.isEmpty()) return;

    sourceImage_tab2 = cv::imread(filename.toStdString());
    if(sourceImage_tab2.empty())
    {
        QMessageBox::warning(this, "Hata", "Kaynak görüntü yüklenemedi!");
        return;
    }

    sourceImagePath_tab2 = filename;

    // Görüntüyü label'a yerleştir
    cv::Mat displayImage = sourceImage_tab2.clone();
    ui->label_source_face_tab_2->setPixmap(
        QPixmap::fromImage(cvMatToQImage(displayImage))
            .scaled(ui->label_source_face_tab_2->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation)
        );

    ui->textBrowser_output_tab_2->append("Kaynak resim yüklendi: " + filename);
}


void MainWindow::on_pushButton_select_target_image_tab_2_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Hedef Resmi Seç", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if(filename.isEmpty()) return;

    targetImage_tab2 = cv::imread(filename.toStdString());
    if(targetImage_tab2.empty())
    {
        QMessageBox::warning(this, "Hata", "Hedef görüntü yüklenemedi!");
        return;
    }

    targetImagePath_tab2 = filename;

    // Görüntüyü label'a yerleştir
    cv::Mat displayImage = targetImage_tab2.clone();
    ui->label_target_face_tab_2->setPixmap(
        QPixmap::fromImage(cvMatToQImage(displayImage))
            .scaled(ui->label_target_face_tab_2->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation)
        );

    ui->textBrowser_output_tab_2->append("Hedef resim yüklendi: " + filename);
}


void MainWindow::on_pushButton_compare_tab_2_clicked()
{
    if(sourceImage_tab2.empty() || targetImage_tab2.empty())
    {
        QMessageBox::warning(this, "Hata", "Lütfen önce kaynak ve hedef resimleri seçin!");
        return;
    }

    compareFaces_tab2();
}


void MainWindow::on_pushButton_clear_tab_2_clicked()
{
    sourceImage_tab2 = cv::Mat();
    targetImage_tab2 = cv::Mat();
    sourceImagePath_tab2.clear();
    targetImagePath_tab2.clear();

    ui->label_source_face_tab_2->clear();
    ui->label_target_face_tab_2->clear();
    ui->textBrowser_output_tab_2->clear();
    ui->progressBar_sim_rate_tab_2->setValue(0);

    ui->label_source_face_tab_2->setText("Kaynak Resim");
    ui->label_target_face_tab_2->setText("Hedef Resim");
}

void MainWindow::compareFaces_tab2()
{
    ui->progressBar_sim_rate_tab_2->setValue(0);
    ui->textBrowser_output_tab_2->clear();
    ui->textBrowser_output_tab_2->append("=== YÜZ KARŞILAŞTIRMA BAŞLATILDI ===");

    try {
        // 1. Adım: Resimleri kontrol et
        ui->textBrowser_output_tab_2->append("\n1. Adım: Resimler kontrol ediliyor...");
        if (sourceImage_tab2.empty() || targetImage_tab2.empty()) {
            ui->textBrowser_output_tab_2->append("Hata: Resimler yüklenemedi!");
            return;
        }
        ui->progressBar_sim_rate_tab_2->setValue(10);

        // 2. Adım: Kaynak resimde yüz tespiti
        ui->textBrowser_output_tab_2->append("\n2. Adım: Kaynak resimde yüz tespiti...");
        std::vector<FaceProposal> faces1 = detectFacesInImage(sourceImage_tab2);
        if (faces1.empty()) {
            ui->textBrowser_output_tab_2->append("Hata: Kaynak resimde yüz bulunamadı!");
            return;
        }
        ui->textBrowser_output_tab_2->append(QString("   - %1 yüz bulundu").arg(faces1.size()));
        ui->progressBar_sim_rate_tab_2->setValue(30);

        // 3. Adım: Hedef resimde yüz tespiti
        ui->textBrowser_output_tab_2->append("\n3. Adım: Hedef resimde yüz tespiti...");
        std::vector<FaceProposal> faces2 = detectFacesInImage(targetImage_tab2);
        if (faces2.empty()) {
            ui->textBrowser_output_tab_2->append("Hata: Hedef resimde yüz bulunamadı!");
            return;
        }
        ui->textBrowser_output_tab_2->append(QString("   - %1 yüz bulundu").arg(faces2.size()));
        ui->progressBar_sim_rate_tab_2->setValue(50);

        // 4. Adım: Gömme vektörlerini çıkar
        ui->textBrowser_output_tab_2->append("\n4. Adım: Yüz özellik vektörleri çıkarılıyor...");
        std::vector<std::vector<float>> embeddings1, embeddings2;

        for (size_t i = 0; i < faces1.size(); i++) {
            ui->textBrowser_output_tab_2->append(QString("   - Kaynak %1. yüz işleniyor...").arg(i + 1));
            std::vector<float> embedding = getFaceEmbedding(sourceImage_tab2, faces1[i].rect);
            if (!embedding.empty()) {
                embeddings1.push_back(embedding);
            }
            ui->progressBar_sim_rate_tab_2->setValue(50 + (i * 10 / faces1.size()));
        }

        for (size_t i = 0; i < faces2.size(); i++) {
            ui->textBrowser_output_tab_2->append(QString("   - Hedef %1. yüz işleniyor...").arg(i + 1));
            std::vector<float> embedding = getFaceEmbedding(targetImage_tab2, faces2[i].rect);
            if (!embedding.empty()) {
                embeddings2.push_back(embedding);
            }
            ui->progressBar_sim_rate_tab_2->setValue(60 + (i * 10 / faces2.size()));
        }

        if (embeddings1.empty() || embeddings2.empty()) {
            ui->textBrowser_output_tab_2->append("Hata: Yüz özellik vektörleri çıkarılamadı!");
            return;
        }
        ui->progressBar_sim_rate_tab_2->setValue(70);

        // 5. Adım: Benzerlik hesapla
        ui->textBrowser_output_tab_2->append("\n5. Adım: Benzerlik hesaplanıyor...");

        float max_similarity = 0.0f;
        int best_pair_i = -1, best_pair_j = -1;
        std::vector<std::vector<float>> similarity_matrix(embeddings1.size(),
                                                          std::vector<float>(embeddings2.size(), 0.0f));

        for (size_t i = 0; i < embeddings1.size(); i++) {
            for (size_t j = 0; j < embeddings2.size(); j++) {
                float similarity = cosineSimilarity(embeddings1[i], embeddings2[j]);
                similarity_matrix[i][j] = similarity;

                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    best_pair_i = i;
                    best_pair_j = j;
                }
            }
            ui->progressBar_sim_rate_tab_2->setValue(70 + (i * 20 / embeddings1.size()));
        }

        ui->progressBar_sim_rate_tab_2->setValue(90);

        // 6. Adım: Sonuçları göster
        ui->textBrowser_output_tab_2->append("\n=== SONUÇLAR ===");

        // Progress bar'ı benzerlik yüzdesine göre ayarla
        int similarity_percentage = (int)(max_similarity * 100);
        ui->progressBar_sim_rate_tab_2->setValue(similarity_percentage);

        ui->textBrowser_output_tab_2->append(QString("En yüksek benzerlik: %1%").arg(similarity_percentage));

        // Benzerlik değerlendirmesi
        QString similarity_comment;

        if (similarity_percentage >= 45) {
            similarity_comment = "%45 Üzeri Benzerlik Aynı Kişi Olma Olasılığı Çok Yüksek.";
        }else {
            similarity_comment = "Aynı Kişi Olma Olasılığı Düşük";
        }


        ui->textBrowser_output_tab_2->append(similarity_comment);

        if (best_pair_i >= 0 && best_pair_j >= 0) {
            ui->textBrowser_output_tab_2->append(QString("\nEşleşen yüzler:"));
            ui->textBrowser_output_tab_2->append(QString("   - Kaynak resim %1. yüz").arg(best_pair_i + 1));
            ui->textBrowser_output_tab_2->append(QString("   - Hedef resim %1. yüz").arg(best_pair_j + 1));
            ui->textBrowser_output_tab_2->append(QString("   - Kaynak yüz güven skoru: %1").arg(faces1[best_pair_i].score, 0, 'f', 3));
            ui->textBrowser_output_tab_2->append(QString("   - Hedef yüz güven skoru: %1").arg(faces2[best_pair_j].score, 0, 'f', 3));
        }

        // Benzerlik matrisini göster
        if (embeddings1.size() > 1 || embeddings2.size() > 1) {
            ui->textBrowser_output_tab_2->append("\nBenzerlik Matrisi:");
            for (size_t i = 0; i < similarity_matrix.size(); i++) {
                QString row_text = QString("Kaynak %1: ").arg(i + 1);
                for (size_t j = 0; j < similarity_matrix[i].size(); j++) {
                    row_text += QString("Hedef%1:%2% ").arg(j + 1).arg((int)(similarity_matrix[i][j] * 100));
                }
                ui->textBrowser_output_tab_2->append(row_text);
            }
        }

        // 7. Adım: Görsel sonuçları hazırla ve göster
        cv::Mat result1 = sourceImage_tab2.clone();
        cv::Mat result2 = targetImage_tab2.clone();

        drawFaceDetectionResult(result1, faces1);
        drawFaceDetectionResult(result2, faces2);

        // En iyi eşleşmeyi vurgula
        if (best_pair_i >= 0 && best_pair_i < faces1.size()) {
            cv::rectangle(result1, faces1[best_pair_i].rect, cv::Scalar(255, 0, 0), 3);
            cv::putText(result1, "EN IYI ESLESME",
                        cv::Point(faces1[best_pair_i].rect.x, faces1[best_pair_i].rect.y - 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        if (best_pair_j >= 0 && best_pair_j < faces2.size()) {
            cv::rectangle(result2, faces2[best_pair_j].rect, cv::Scalar(255, 0, 0), 3);
            cv::putText(result2, "EN IYI ESLESME",
                        cv::Point(faces2[best_pair_j].rect.x, faces2[best_pair_j].rect.y - 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }

        // Label'lara işlenmiş resimleri yerleştir
        ui->label_source_face_tab_2->setPixmap(
            QPixmap::fromImage(cvMatToQImage(result1))
                .scaled(ui->label_source_face_tab_2->size(),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation)
            );

        ui->label_target_face_tab_2->setPixmap(
            QPixmap::fromImage(cvMatToQImage(result2))
                .scaled(ui->label_target_face_tab_2->size(),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation)
            );

        ui->progressBar_sim_rate_tab_2->setValue(similarity_percentage);
        ui->textBrowser_output_tab_2->append("\n✅ İşlem tamamlandı!");

    } catch (const std::exception& e) {
        ui->textBrowser_output_tab_2->append(QString("Hata: %1").arg(e.what()));
        ui->progressBar_sim_rate_tab_2->setValue(0);
    }
}
