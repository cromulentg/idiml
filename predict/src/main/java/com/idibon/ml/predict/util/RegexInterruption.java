package com.idibon.ml.predict.util;

/**
 * Class for being thrown from a RegularExpression that
 * took too long to run.
 */
public class RegexInterruption extends RuntimeException {

    private int backtracks;
    private long duration;

    public RegexInterruption(int backtracks, long duration) {
        this.backtracks = backtracks;
        this.duration = duration;
    }

    public int getBacktracks() {
        return backtracks;
    }

    public long getDuration() {
        return duration;
    }
}
