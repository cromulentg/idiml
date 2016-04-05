package com.idibon.ml.feature.tokenizer

import java.text.CharacterIterator
import com.ibm.icu.text.BreakIterator

/** State machine for parsing HTML markup elements
  *
  * Builds upon XML tokenizer, adding special handling for the opaque (CDATA-
  * like) sections used by inline <script> and <style> elements, so that the
  * inline Javascript and CSS code is also treated as markup.
  */
class HTMLBreakIterator(delegate: BreakIterator)
    extends XMLBreakIterator(delegate) {

  // stores the previous returned boundary
  private[this] var _lastBoundary: Int = -1
  // caches the actual text being analyzed
  private[this] var _rawText: String = null

  /** Returns the position of the next boundary
    *
    * {@link com.ibm.icu.text.BreakIterator#next}
    */
  override def next: Int = {
    val nextBoundary = super.next
    val isMarkup = getRuleStatus() == Tag.ruleStatus(Tag.Markup)

    if (_lastBoundary != -1 && isMarkup) {
      /* call following() to make sure that the delegate break iterator is
       * actually advanced to a boundary position at or past the detected
       * markup boundary */
      _lastBoundary = following(handleInlineCode(_lastBoundary, nextBoundary) - 1)
    } else {
      // no special handling needed, not a markup section
      _lastBoundary = nextBoundary
    }
    _lastBoundary
  }

  /** Moves to and returns the first boundary in the text
    *
    * {@link com.ibm.icu.text.BreakIterator#first}
    */
  override def first: Int = {
    _lastBoundary = super.first
    _lastBoundary
  }

  /** Sets the iterator to analyze a new piece of text.
    *
    * {@link com.ibm.icu.text.BreakIterator#setText}
    * @param text the text to analyze
    */
  override def setText(text: String) {
    _lastBoundary = -1
    _rawText = text
    super.setText(text)
  }

  override def setText(text: java.text.CharacterIterator) {
    _lastBoundary = -1
    _rawText = null
    super.setText(text)
  }

  /** Expands markup tokens for tags which include inline code elements
    *
    * Treats <script> and <style> sections as one long markup element
    * from the open tag to the end tag
    *
    * @param start start position of the current token
    * @param end end position of the current token
    * @return modified end position for the boundary
    */
  private[this] def handleInlineCode(start: Int, end: Int): Int = {
    if (end <= start) return end
    if (_rawText == null) _rawText = HTMLBreakIterator.getString(getText())

    if (HTMLBreakIterator.isSelfClosedTag(_rawText, end)) {
      /* self-closing tags aren't valid HTML, but they're (incorrectly) used
       * often enough for script and style resources that they need to be
       * detected and treated as token boundaries so as to not under-
       * tokenize the document */
      end
    } else if (HTMLBreakIterator.isScriptTag(_rawText, start)) {
      /* scan for the </script> end tag; start the scan from the original
       * "end" boundary position, which marks the close delimiter for the
       * <script> tag open delimiter */
      val close = HTMLBreakIterator.indexOfEndTag(_rawText, end,
        HTMLBreakIterator.SCRIPT_TAG)
      /* the boundary is 1 character after the '>' character, if present;
       * if there is no '>' character, return the original boundary */
      if (close == -1) end else close + 1
    } else if (HTMLBreakIterator.isStyleTag(_rawText, start)) {
      // scan for the </style> end tag
      val close = HTMLBreakIterator.indexOfEndTag(_rawText, end,
        HTMLBreakIterator.STYLE_TAG)
      // the boundary is 1 character after the '>' character, if present
      if (close == -1) end else close + 1
    } else {
      // no special handling required, just return the original boundary
      end
    }
  }
}

/** Companion object for HTMLBreakIterator */
private[tokenizer] object HTMLBreakIterator {

  val SCRIPT_TAG = "script"
  val STYLE_TAG = "style"

  /** Converts a CharacterIterator into a String */
  def getString(chars: CharacterIterator): String = {
    val index = chars.getIndex
    val result = new StringBuilder
    result.append(chars.first)
    Stream.continually(chars.next)
      .takeWhile(_ != CharacterIterator.DONE)
      .foreach(ch => result.append(ch))
    chars.setIndex(index)
    result.toString
  }

  /** Returns the text index of the boundary following a matched end tag
    *
    * @param text text to analyze
    * @param fromIndex starting position to scan
    * @param tag expected end tag
    * @return index of the close of the end tag, or -1 if not found
    */
  def indexOfEndTag(text: String, fromIndex: Int, tag: String): Int = {
    var etago = fromIndex - 1
    do {
      etago = text.indexOf("</", etago + 1)
      /* if there is an etago (end-tag open) sequence, see if it matches
       * the expected tag; if so, find the tag close character. since
       * attributes aren't valid within end tags, the scan for the tag
       * close can perform a dumb character match */
      if (etago != -1 && isTag(text, etago, "</", tag))
        return text.indexOf('>', etago + 2 + tag.length)
    } while (etago != -1)
    // no matched end tag found, abort
    -1
  }

  /** Returns true if a specific start tag is at an offset in the text
    *
    * Performs a case-insensitive comparison between the tag name and the
    * document text to determine if the specified start tag exists at the
    * location in the text.
    *
    * @param text text to analyze
    * @param offs location within the text to compare
    * @param pre tag prefix (i.e., the '<' or '</' sequence)
    * @param tag tag to check
    * @return true if the tag exists at the position, else false
    */
  def isTag(text: String, offs: Int, pre: String, tag: String): Boolean = {
    (text.length > offs + pre.length + tag.length &&
      text.regionMatches(offs, pre, 0, pre.length) &&
      text.regionMatches(true, offs + pre.length, tag, 0, tag.length) &&
      (XMLBreakIterator.isWhitespace(text.charAt(offs + pre.length + tag.length)) ||
        (text.charAt(offs + pre.length + tag.length) == '>')))
  }

  /** Returns true if a <script> start tag exists at the offset
    *
    * @param text text to analyze
    * @param offs starting position to check for a script tag
    * @return if a script tag exists at the position
    */
  def isScriptTag(text: String, offs: Int) = isTag(text, offs, "<", SCRIPT_TAG)

  /** Returns true if a <style> start tag exists at the offset
    *
    * @param text text to analyze
    * @param offs starting position to check for a script tag
    * @return if a script tag exists at the position
    */
  def isStyleTag(text: String, offs: Int) = isTag(text, offs, "<", STYLE_TAG)

  /** Returns true if a self-closed tag ends at the specified offset
    *
    * @param t text to analyze
    * @param e boundary following a detected tag
    * @return true if the tag was self-closing, else false
    */
  def isSelfClosedTag(t: String, e: Int) = e >= 2 && t.charAt(e - 2) == '/'
}
