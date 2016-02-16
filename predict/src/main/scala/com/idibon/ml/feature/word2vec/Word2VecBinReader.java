package com.idibon.ml.feature.word2vec;

import java.io.*;
import java.net.URI;
import java.util.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPInputStream;

/**
 * Class for reading word2vec gzipped binary vector files and outputting Java hashmaps
 */
public class Word2VecBinReader {

    private long words;
    private long size;

    /**
     * Converts an ASCII string long integer into a Java long integer
     *
     * @param dis DatatInputStream containing a long integer
     * @return the actual long integer
     */
    public long readAsciiLong(DataInputStream dis) {
        String accum = "";
        while(true){
            try{
                char c = (char) dis.readByte();
                if (Character.isWhitespace(c)) {
                    break;
                }
                accum += c;
            } catch (IOException e){
                e.printStackTrace();
            }
        }
        return Long.parseLong(accum);
    }

    /**
     * Parses a gzipped binary file output by the word2vec C implementation
     *
     * @param uri uri to gzipped bin file
     * @return Map of words to vectors
     */
    public Map<String, float[]> parseBinFile(URI uri) {
        Map<String, float[]> vectors = new HashMap<>();

        try (DataInputStream data = new DataInputStream(
                new GZIPInputStream(new FileInputStream(new File(uri)), 65536))) {

            words = readAsciiLong(data);
            size = readAsciiLong(data);

            if (size > Integer.MAX_VALUE)
                throw new IllegalArgumentException("Too many dimensions");

            for (long b = 0; b < words; b++) {
                ByteArrayOutputStream wordBuffer = new ByteArrayOutputStream();
                for (int c = data.readByte(); c != -1 && c != ' '; c = data.readByte())
                    wordBuffer.write(c);

                String word = new String(wordBuffer.toByteArray(), "UTF-8");
                float[] vector = new float[(int) size];
                /* the original implementation generally runs on little-endian
                 * native architectures, and there is no accommodation for big-
                 * endian in the file format, so just assume that an endian-
                 * swap is necessary to get the float values into Java */
                byte[] swap = new byte[4];
                ByteBuffer endianConverter = ByteBuffer.wrap(swap).order(ByteOrder.LITTLE_ENDIAN);

                for (int i = 0; i < (int) size; i++) {
                    data.readFully(swap);
                    vector[i] = endianConverter.getFloat();
                    endianConverter.rewind();
                }

                vectors.put(word.trim(), vector);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return vectors;
    }

}
