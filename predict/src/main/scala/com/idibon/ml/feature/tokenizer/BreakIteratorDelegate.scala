package com.idibon.ml.feature.tokenizer

import java.text.CharacterIterator
import com.ibm.icu.text.{BreakIterator, UCharacterIterator}

/** Abstract base class for BreakIterator decorator classes
  *
  * Sub-classes are expected to wrap an underlying break iterator with
  * custom logic that suppresses certain types of segment boundaries that
  * aren't detected by the underlying iterator.
  *
  * Implements all of the abstract methods in the BreakIterator class
  * except for the basic next method; however, most of the more complicated
  * are implemented by delegating directly to the underlying iterator, so
  * these methods will not be aware of any breaks that would have been
  * suppressed.
  */
private[tokenizer] abstract class BreakIteratorDelegate(delegate: BreakIterator)
    extends BreakIterator {

  // cache the character iterator across boundaries, only reset for new text
  protected var _characters: UCharacterIterator = null

  /** Returns a mutable character iterator for analyzing the current text
    *
    * Adapted from {@link com.ibm.icu.impl.SimpleFilteredSentenceBreakIterator}
    */
  protected def getCharacters(): UCharacterIterator = {
    val base = BreakIteratorDelegate.publicClone.invoke(delegate.getText())
    val baseIt = base.asInstanceOf[CharacterIterator]
    UCharacterIterator.getInstance(baseIt)
  }

  /** Move the iterator forward or backward the specified number of boundaries
    *
    * {@link com.ibm.icu.text.BreakIterator#next(int)
    */
  def next(count: Int): Int = delegate.next(count)

  /** Returns the current position in the analyzed text
    *
    * {@link com.ibm.icu.text.BreakIterator#current}
    */
  def current: Int = delegate.current

  /** Moves to the first boundary position in the analyzed text
    *
    * {@link com.ibm.icu.text.BreakIterator#first}
    */
  def first: Int = delegate.first

  /** Moves to the last boundary position in the analyzed text
    *
    * {@link com.ibm.icu.text.BreakIterator#last}
    */
  def last: Int = delegate.last

  /** Move the iterator to the previous boundary
    *
    * {@link com.ibm.icu.text.BreakIterator#previous}
    */
  def previous: Int = delegate.previous

  /** Move to the first boundary following the specified position
    *
    * {@link com.ibm.icu.text.BreakIterator#following}
    */
  def following(after: Int): Int = delegate.following(after)

  /** Returns a {@link CharacterIterator} over the current text
    *
    * {@link com.ibm.icu.text.BreakIterator#getText}
    */
  def getText: CharacterIterator = delegate.getText

  /** Sets the iterator to analyze a new piece of text.
    *
    * {@link com.ibm.icu.text.BreakIterator#setText}
    * @param text the text to analyze
    */
  override def setText(text: String) {
    delegate.setText(text)
    _characters = null
  }

  def setText(text: CharacterIterator) {
    delegate.setText(text)
    _characters = null
  }

}

/** Companion to BreakIteratorDelegate */
private[this] object BreakIteratorDelegate {
  /* work-around Scala bug SI-6760, where public interface declarations
   * don't override protected implementations in super-classes. normally
   * not a problem, except for the known-awful Java Cloneable interface */
  val publicClone = classOf[CharacterIterator].getDeclaredMethod("clone")
}
