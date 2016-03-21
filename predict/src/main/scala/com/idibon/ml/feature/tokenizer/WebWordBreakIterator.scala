package com.idibon.ml.feature.tokenizer

import scala.collection.mutable.HashSet
import scala.util.Try
import java.io.{BufferedReader, InputStreamReader}
import java.text.CharacterIterator

import com.idibon.ml.feature.tokenizer.Unicode._

import com.ibm.icu.text.{BreakIterator, FilteredBreakIteratorBuilder,
  UCharacterIterator, UForwardCharacterIterator}
import com.ibm.icu.util.{BytesTrie, CharsTrie, CharsTrieBuilder,
  StringTrieBuilder, ULocale}
import com.ibm.icu.lang.UCharacter

import com.typesafe.scalalogging.StrictLogging

/** BreakIterator decorator class for tokenizing "words" in a web context
  *
  * Suppresses word breaks from a delegate BreakIterator using a dictionary
  * of special terms (e.g., emoticons) and parser state machines for other
  * strictly-defined types (e.g., URIs), so that the terms are returned as
  * atomic units with a single boundary, rather than detecting numerous
  * intra-URI boundaries
  */
class WebWordBreakIterator(delegate: BreakIterator) extends BreakIterator {

  // cache the character iterator across boundaries
  private [this] var _characters: UCharacterIterator = null

  // shallow-copy the trie of special boundaries
  private [this] val trie = WebWordBreakIterator.TRIE.clone

  // URI parser state machine
  private [this] val uriState = new URIStateMachine

  /* work-around Scala bug SI-6760, where public interface declarations
   * don't override protected implementations in super-classes. normally
   * not a problem, except for the known-awful Java Cloneable interface */
  private [this] val publicClone =
    classOf[CharacterIterator].getDeclaredMethod("clone")

  /** Returns the position of the next boundary
    *
    * {@link com.ibm.icu.text.BreakIterator}
    */
  def next: Int = {
    if (_characters == null) _characters = getCharacters()
    _characters.setIndex(current)
    /* since BreakIterator#following returns the first boundary after the
     * provided codepoint, we call it at the last codepoint detected
     * within our special elements (URIs, dictionary matches, etc.) to
     * properly advance the delegate's state to a boundary at or past the
     * boundary detected in this class */
    val result = trie.matches(_characters).flatMap(_ match {
      case (TrieResult.URI_HIERARCHICAL, boundary) =>
        hierUriMatch(boundary).map(b => delegate.following(b - 1))
      case (TrieResult.URI_OPAQUE, boundary) =>
        opaqueUriMatch(boundary).map(b => delegate.following(b - 1))
      case (TrieResult.HTML_DECIMAL_REF, boundary) =>
        htmlRefMatch(boundary, false).map(b => delegate.following(b - 1))
      case (TrieResult.HTML_HEXADECIMAL_REF, boundary) =>
        htmlRefMatch(boundary, true).map(b => delegate.following(b - 1))
      case (_, boundary) =>
        Some(delegate.following(boundary - 1))
    }).getOrElse(delegate.next)

    // make the character iterator collectable if this is the last boundary
    if (result == BreakIterator.DONE) _characters = null
    result
  }

  /** Returns a mutable character iterator for analyzing the current text
    *
    * Adapted from {@link com.ibm.icu.impl.SimpleFilteredSentenceBreakIterator}
    */
  private [this] def getCharacters(): UCharacterIterator = {
    val base = publicClone.invoke(delegate.getText())
    UCharacterIterator.getInstance(base.asInstanceOf[CharacterIterator])
  }

  /** Returns the boundary of an HTML hexadecimal character reference
    *
    * Hex references have the form &#x{hex digits};, forming any valid
    * Unicode codepoint.
    *
    * Matching starts after the leading x.
    *
    * @param start position (in UTF-16 code units) to start hex match in string
    * @return the boundary of the reference, if a reference is detected, or None
    */
  private [tokenizer] def htmlRefMatch(start: Int, hex: Boolean): Option[Int] = {
    _characters.setIndex(start)
    var last = UForwardCharacterIterator.DONE
    val predicate = if (hex) isAsciiHexDigit _ else isAsciiDigit _
    val value = Stream.continually({ last = _characters.nextCodePoint; last })
      .takeWhile(predicate)
      .foldLeft(0)((sum, hex) => sum * 16 + Character.getNumericValue(hex))
    if (last == ';' && value <= Character.MAX_CODE_POINT)
      Some(_characters.getIndex())
    else
      None
  }

  /** Returns the boundary of an RFC-3987 hierarchical IRI */
  private [tokenizer] def hierUriMatch(start: Int): Option[Int] = {
    _characters.setIndex(start)
    // parsing will begin at the authority component
    uriState.reset(URIStateMachineState.AUTHORITY)
    /* keep track of the last codepoint extracted from the stream, to determine
     * if the match ended because of end-of-string or due to an illegal
     * character. in the case of illegal characters, rewind the stream to the
     * previous (valid) codepoint and declare it the boundary. this properly
     * accounts for codepoints with either 1 or 2 code units */
    var last = UForwardCharacterIterator.DONE
    Stream.continually({ last = _characters.nextCodePoint; last })
      .takeWhile(ch => uriState.nextCodePoint(ch).hasNext)
      .force
    /* the loop above will consume up to the first invalid character;
     * if the URI matched, return the last index inside the URI as
     * the boundary */
    if (last != UForwardCharacterIterator.DONE) _characters.previousCodePoint
    if (uriState.hasMatch) {
      /* since end-of-sentence punctuation is often very important for NLP
       * models, rather than greedily tokenizing  trailing punctuation with
       * the URI, assume that the punctuation was intended to mark the end-
       * of sentence and tokenize it separately. allow trailing solidus
       * characters, though, since these are almost always intended to be
       * part of the URL */
      backupWhile({
        val cp = _characters.currentCodePoint
        (cp != '/' && isPunctuation(cp)) || isGraphemeExtender(cp)
      })
      Some(_characters.getIndex())
    } else {
      None
    }
  }

  /** Returns the boundary of an RFC-2396 opaque URI */
  private [tokenizer] def opaqueUriMatch(start: Int): Option[Int] = {
    _characters.setIndex(start)
    // parsing begins at the opaque component
    uriState.reset(URIStateMachineState.OPAQUE)
    var last = UForwardCharacterIterator.DONE
    Stream.continually({ last = _characters.nextCodePoint; last })
      .takeWhile(ch => uriState.nextCodePoint(ch).hasNext)
      .force
    if (last != UForwardCharacterIterator.DONE) _characters.previousCodePoint
    if (uriState.hasMatch) {
      // see comment above about end-of-sentence punctuation
      backupWhile({
        val cp = _characters.currentCodePoint
        isPunctuation(cp) || isGraphemeExtender(cp)
      })
      Some(_characters.getIndex())
    } else {
      None
    }
  }

  /** Traverses backwards through the string while a condition is true
    *
    * This can be used to insert boundaries within segments of text that are
    * greedily tokenized, like URIs
    */
  private [this] def backupWhile(p: => Boolean) = {
    while (_characters.previousCodePoint != UForwardCharacterIterator.DONE && p) {}
    if (!p) {
      /* move forward one full grapheme after terminating on the first
       * codepoint that fails the predicate function */
      Stream.continually(_characters.nextCodePoint)
        .takeWhile(cp => isGraphemeExtender(cp))
        .force
    }
  }

  /** Move the iterator forward or backward the specified number of boundaries
    *
    * {@link com.ibm.icu.text.BreakIterator#next(int)
    */
  def next(count: Int): Int = ???

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
  def previous: Int = ???

  /** Move to the first boundary following the specified position
    *
    * {@link com.ibm.icu.text.BreakIterator#following}
    */
  def following(after: Int): Int = ???

  /** Returns a {@link java.text.CharacterIterator} over the current text
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

/** Maintain tries for forward- and reverse-indexed text */
private[tokenizer] case class Trie(trie: CharsTrie) {
  override def clone: Trie = Trie(trie.clone.asInstanceOf[CharsTrie])

  /** Compares the current text position against values in the trie
    *
    * Returns the type and length of the matched text if a match is detected.
    *
    * @param text iterator to analyze the current text boundary
    * @return match result, if a match is detected
    */
  def matches(text: UCharacterIterator): Option[(TrieResult.Value, Int)] = {
    trie.reset()
    val x: Option[(TrieResult.Value, Int)] = None
    Stream.continually(text.nextCodePoint)
      .takeWhile(u => trie.nextForCodePoint(u).hasNext())
      .foldLeft(x)({ case (best, codePoint) => trie.current.hasValue() match {
        case true => Some(TrieResult(trie.getValue()), text.getIndex())
        case _ => best
      }})
  }
}

/** Various match states stored with nodes in the CharsTrie */
object TrieResult extends Enumeration {
  val EMOTICON,            // Emoticon (exact match)
    HTML_NAMED_REF,        // Named character reference (exact match)
    HTML_DECIMAL_REF,      // Decimal character reference (prefix match)
    HTML_HEXADECIMAL_REF,  // Hexadecimal character reference (prefix match)
    URI_HIERARCHICAL,      // Hierarchical (RFC-3987) URI
    URI_OPAQUE = Value     // Opaque (RFC-2396) URI
}

/** Companion object for WebWordBreakIterator */
object WebWordBreakIterator extends StrictLogging {

  val TRIE: Trie = {
    val builder = new CharsTrieBuilder()
    val words = HashSet[String]()

    def addTrieEntry(str: String, result: TrieResult.Value) {
      if (words.add(str)) builder.add(str, result.id)
    }

    // initialize the trie with some URI schemes
    addTrieEntry("http://", TrieResult.URI_HIERARCHICAL)
    addTrieEntry("https://", TrieResult.URI_HIERARCHICAL)
    addTrieEntry("file://", TrieResult.URI_HIERARCHICAL)
    addTrieEntry("ftp://", TrieResult.URI_HIERARCHICAL)
    addTrieEntry("mailto:", TrieResult.URI_OPAQUE)
    addTrieEntry("tel:", TrieResult.URI_OPAQUE)
    addTrieEntry("&#x", TrieResult.HTML_HEXADECIMAL_REF)
    addTrieEntry("&#", TrieResult.HTML_DECIMAL_REF)

    readStrings("data/emoticons.txt").foreach(str => {
      addTrieEntry(str, TrieResult.EMOTICON)
    })

    readStrings("data/character_refs.txt").foreach(str => {
      // split into the entity name and code point(s)
      val splitPoint = str.indexOf(';')
      /* ignore the code points in the table for now; normalizing to
       * code points probably belongs in BagOfWords transform */
      addTrieEntry(str.substring(0, splitPoint + 1), TrieResult.HTML_NAMED_REF)
    })

    Trie(builder.build(StringTrieBuilder.Option.FAST))
  }

  /** Reads each line of UTF-8 encoded text file into a set of strings */
  def readStrings(resource: String): HashSet[String] = Try({
    val r = getClass().getClassLoader().getResourceAsStream(resource)
    val set = HashSet[String]()
    try {
      val reader = new BufferedReader(new InputStreamReader(r, "UTF-8"))
      Stream.continually(reader.readLine())
        .takeWhile(_ != null)
        .foreach(set += _.trim())
    } finally {
      r.close()
    }
    set
  }).recover({ case (error) => {
    logger.error(s"Error reading $resource", error)
    HashSet[String]()
  }}).get
}
