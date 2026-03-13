import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import com.rk3568.face 1.0

ApplicationWindow {
    id: rootWindow
    visible: true
    title: "RK3568 人脸识别系统"

    readonly property int sidebarWidth: 200
    readonly property int btnHeight: 80
    readonly property int pad: 12

    // ===== 侧边栏 =====
    Rectangle {
        id: sidebar
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        width: sidebarWidth
        color: "#E3F2FD"

        Text {
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.topMargin: 24
            text: "人脸识别"
            font.pixelSize: 20
            font.bold: true
            color: "#2196F3"
        }

        Text {
            anchors.centerIn: parent
            text: "数据库管理\n(即将推出)"
            font.pixelSize: 14
            color: "#757575"
            horizontalAlignment: Text.AlignHCenter
        }

        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 12
            anchors.bottomMargin: 16
            height: 50
            color: "white"
            radius: 6
            border.color: "#BBDEFB"
            border.width: 1

            Row {
                anchors.centerIn: parent
                spacing: 6

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: backend.cameraReady ? "#4CAF50" : "#F44336"
                    anchors.verticalCenter: parent.verticalCenter
                }
                Text {
                    text: backend.cameraReady ? "摄像头就绪" : "未连接"
                    font.pixelSize: 13
                    font.bold: true
                    color: backend.cameraReady ? "#4CAF50" : "#F44336"
                }
            }
        }
    }

    // ===== 摄像头区域 =====
    Rectangle {
        id: cameraArea
        anchors.top: parent.top
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.bottom: buttonRow.top
        anchors.margins: pad
        anchors.bottomMargin: pad / 2
        color: "#1A1A1A"
        radius: 8
        border.color: "#444444"
        border.width: 1

        VideoFrameItem {
            id: videoFrame
            objectName: "videoFrame"
            anchors.fill: parent
        }

        // 状态提示
        Rectangle {
            id: statusHint
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.topMargin: 12
            width: statusText.implicitWidth + 32
            height: 40
            radius: 20
            color: statusColor
            visible: backend.statusMessage.length > 0

            property color statusColor: "#4CAF50"

            Connections {
                target: backend
                function onStatusMessageChanged() {
                    var msg = backend.statusMessage
                    if (msg.indexOf("成功") >= 0 || msg.indexOf("识别") >= 0)
                        statusHint.statusColor = "#4CAF50"
                    else if (msg.indexOf("失败") >= 0 || msg.indexOf("错误") >= 0)
                        statusHint.statusColor = "#F44336"
                    else
                        statusHint.statusColor = "#FF9800"
                }
            }

            Text {
                id: statusText
                anchors.centerIn: parent
                text: backend.statusMessage
                font.pixelSize: 16
                font.bold: true
                color: "white"
            }
        }
    }

    // ===== 底部按钮 =====
    Row {
        id: buttonRow
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: pad
        anchors.leftMargin: pad
        height: btnHeight
        spacing: pad

        ModernButton {
            width: (parent.width - pad) / 2
            height: parent.height
            text: "录入人脸"
            buttonColor: "#4CAF50"
            enabled: backend.cameraReady && !backend.isProcessing
            onClicked: backend.enrollFace()
        }

        ModernButton {
            width: (parent.width - pad) / 2
            height: parent.height
            text: "识别人脸"
            buttonColor: "#2196F3"
            enabled: backend.cameraReady && !backend.isProcessing
            onClicked: backend.recognizeFace()
        }
    }
}
