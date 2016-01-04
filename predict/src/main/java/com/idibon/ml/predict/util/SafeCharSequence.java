package com.idibon.ml.predict.util;

/**
 * This helper class implements enough of the CharSequence interface
 * to be used by regex.Matcher, and also keeps track of how many times
 * the matcher has backtracked, so that we can kill regexes that take
 * too long to run.
 * <p>
 * This utterly ridiculous but effective approach is based on this article:
 * http://ocpsoft.org/regex/how-to-interrupt-a-long-running-infinite-java-regular-expression/
 * (Based on the JRuby class we implement in idisage/base.rb)
 */
public class SafeCharSequence implements CharSequence {

    /**
     * FIXME: These three predictions parameters are hard-coded here as
     * constants but they really should be settable as parameters to the actual
     * `predict` method.
     *
     * Max # of times a dictionary regex can backtrack before we kill it.
     * In the unit tests on a m3.2xlarge, it takes about 100-200ms for a regex
     * to use up 50k backtracks.
    */
    public static final int MAX_REGEX_BACKTRACKS = 25000;

    private String sequence;
    private int backtrackLimit;
    private int backtracks;
    private int lastIndex;
    private long startTime;

    /**
     * Constructor.
     * @param sequence The string we want to control the regex over.
     * @param backtrackLimit The backtrack limit to enforce.
     */
    public SafeCharSequence(String sequence, int backtrackLimit) {
        this.sequence = sequence;
        this.backtrackLimit = backtrackLimit;
        this.backtracks = this.backtrackLimit;
        this.lastIndex = -1;
        this.startTime = System.currentTimeMillis();
    }

    @Override public int length() {
        return this.sequence.length();
    }

    /**
     * This method checks how many backtracks happen and throws a
     * runtime exception if we exceed the number of backtracks.
     * @param index
     * @return
     */
    @Override public char charAt(int index) {
        /*
         * Are we being asked for an earlier character than last time? If so, we
         * assume that the regex matcher has backtracked, and we decrement our
         * backtrack counter. If it reaches zero, we raise an exception and
         * bail out of the matcher.
         */
        if (index < this.lastIndex) {
            this.backtracks -= 1;
            if (this.backtracks <= 0) {
                long duration = System.currentTimeMillis() - this.startTime;
                throw new RegexInterruption(this.backtracks, duration);
            }
        }
        this.lastIndex = index;
        return sequence.charAt(index);
    }

    @Override public CharSequence subSequence(int start, int end) {
        return new SafeCharSequence(sequence.substring(start, end), this.backtrackLimit);
    }

    @Override public String toString() {
        return this.sequence;
    }
}
