import QtQuick 2.15
import QtQuick.Controls 2.15

Button {
    id: control

    property color buttonColor: "#2196F3"

    background: Rectangle {
        color: control.down   ? Qt.darker(control.buttonColor, 1.15) :
               control.hovered ? Qt.lighter(control.buttonColor, 1.1) :
                                  control.buttonColor
        radius: 10
        opacity: control.enabled ? 1.0 : 0.45

        Behavior on color {
            ColorAnimation { duration: 120 }
        }

        // 底部描边模拟立体感（代替 DropShadow）
        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: 3
            radius: 10
            color: Qt.darker(control.buttonColor, 1.3)
            opacity: control.enabled ? 0.5 : 0
            visible: !control.down
        }

        // 按压缩放
        scale: control.down ? 0.97 : 1.0
        Behavior on scale {
            NumberAnimation { duration: 80 }
        }
    }

    contentItem: Text {
        text: control.text
        font.pixelSize: 22
        font.bold: true
        color: "white"
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
