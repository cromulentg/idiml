package com.idibon.ml.app;

import java.awt.*;
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.BorderLayout;
import javax.swing.JLabel;
import javax.swing.ImageIcon;

/**
 * Easter egg class :D
 *
 * See https://idibon.slack.com/archives/spark-ml/p1453402061000048
 */
public class WiggleWiggle extends JFrame implements Runnable {
    JPanel contentPane;
    JLabel imageLabel = new JLabel();
    JLabel headerLabel = new JLabel();

    public WiggleWiggle() {

    }

    public void terminate() {
        this.dispose();
    }

    public static void main(String[] args) {
        new WiggleWiggle();
    }

    @Override public void run() {
        try {
            setDefaultCloseOperation(DISPOSE_ON_CLOSE);
            contentPane = (JPanel) getContentPane();
            contentPane.setLayout(new BorderLayout());
            setSize(new Dimension(400, 200));
            setTitle("Oooo we're training!!!!");
            // add the header label
            headerLabel.setFont(new java.awt.Font("Comic Sans MS", Font.BOLD, 16));
            // TODO: some random quotes
            // headerLabel.setText("YOUR QUOTE HERE");
            contentPane.add(headerLabel, java.awt.BorderLayout.NORTH);
            // add the image label
            ImageIcon ii = new ImageIcon(this.getClass().getResource("/wiggle-wiggle.gif"));
            imageLabel.setIcon(ii);
            contentPane.add(imageLabel, java.awt.BorderLayout.CENTER);
            // show it
            this.setLocationRelativeTo(null);
            this.setVisible(true);
        } catch (Exception exception) {
            exception.printStackTrace();
            this.dispose();
        }
    }
}
