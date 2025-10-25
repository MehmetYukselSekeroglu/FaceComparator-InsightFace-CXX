#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <QMainWindow>
#include <QString>
#include <QImage>
#include <QPixmap>
#include <QLabel>

struct FaceProposal
{
    cv::Rect rect;
    float score;
    std::vector<cv::Point2f> landmarks;
};

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_detect_tab_1_clicked();
    void on_pushButton_selectFile_tab_1_clicked();
    void on_pushButton_select_source_image_tab_2_clicked();
    void on_pushButton_select_target_image_tab_2_clicked();
    void on_pushButton_compare_tab_2_clicked();
    void on_pushButton_clear_tab_2_clicked();

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    Ui::MainWindow *ui;

    // ONNX Runtime environment ve session
    Ort::Env env;
    Ort::Session* detector;
    Ort::Session* embedder;

    // Modellerin yolu
    std::string model_detection;
    std::string model_embedding;

    // Yardımcı fonksiyonlar
    cv::Mat read_image(const QString &path);
    std::vector<float> extract_embedding(const cv::Mat &face);
    cv::Mat currentImage;
    cv::Mat original_image_for_analysis;

    QImage cvMatToQImage(const cv::Mat &mat);
    void updateImageLabel(QLabel* label, const QString& imagePath);
    void refreshAllImages();

    QString sourceImagePath_tab2;
    QString targetImagePath_tab2;
    cv::Mat sourceImage_tab2;
    cv::Mat targetImage_tab2;

    // **EKSİK FONKSİYONLARI BURAYA EKLE**
    float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
    void performNMS(std::vector<FaceProposal>& proposals, float iou_threshold, std::vector<FaceProposal>& final_faces);

    // Karşılaştırma fonksiyonları
    void compareFaces_tab2();
    std::vector<FaceProposal> detectFacesInImage(const cv::Mat& image);
    std::vector<float> getFaceEmbedding(const cv::Mat& image, const cv::Rect& faceRect);
    float cosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2);
    void drawFaceDetectionResult(cv::Mat& image, const std::vector<FaceProposal>& faces);



    // YENİ: Label içeriklerini tutacak member'lar
    QPixmap m_pixmapTab1;
    QPixmap m_pixmapTab2Source;
    QPixmap m_pixmapTab2Target;

    // YENİ: Pixmap'leri alıp label'lara ölçekleyerek basan yardımcılar
    void updateTab1Pixmap();
    void updateTab2SourcePixmap();
    void updateTab2TargetPixmap();

    void setAboutText(void);
};
#endif // MAINWINDOW_H
