package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.tokenizer.Unicode._

import com.ibm.icu.lang.UCharacter
import com.ibm.icu.text.{BreakIterator, UCharacterIterator,
  UForwardCharacterIterator}
import com.ibm.icu.util.{CharsTrieBuilder, StringTrieBuilder}

/** State machine for parsing XML markup elements
  *
  * Identifies XML tags and comments, but does not interpret the resulting
  * elements, or verify that a valid DOM is formed by them. Because of
  * the lack of interpretation, this class may also be used for identifying
  * HTML tags.
  *
  * Loosely follows the XML specification: CDATA sections are tokenized as
  * document content and most of the possible markup variations (e.g., nested
  * conditional sections and inline declarations) should be tokenized as
  * individual units to the actual terminator. However, the parsing is not
  * context-sensitive, so this parser also accepts various invalid sequences
  * (e.g., attributes within an end-of-tag marker), inline entities are
  * expanded or treated specially, and it's entirely possible that some edge
  * cases allowed in the specification are not handled correctly.
  */
class XMLBreakIterator(delegate: BreakIterator)
    extends BreakIteratorDelegate(delegate) {

  // shallow-copy the trie of XML element types
  private[this] val trie = XMLBreakIterator.TRIE.clone

  // keep track of the current text processing mechanism within XML
  private[this] var _state: XMLParserState.Value = XMLParserState.CharacterData

  // stores the type of break that was detected, for getRuleStatus
  private[this] var _lastRuleStatus: Int = 0

  /** Returns the position of the next boundary
    *
    * {@link com.ibm.icu.text.BreakIterator}
    */
  def next: Int = {
    if (_characters == null) _characters = getCharacters()
    _characters.setIndex(current)
    val result = _state match {
      case XMLParserState.CharacterData => nextCharacterData
      case XMLParserState.CDATA => nextCDATA
    }
    // make the character iterator collectable after the last boundary
    if (result == BreakIterator.DONE) _characters = null
    result
  }

  /** Locates the next boundary within intermingled character data / markup
    *
    * Looks for start-of-markup indicators and parses inline markup as-needed;
    * otherwise tokenizes character data using the provided delegate.
    */
  private[this] def nextCharacterData: Int = {
    trie.matches(_characters).map(element => {
      _lastRuleStatus = Tag.ruleStatus(Tag.Markup)
      element match {
      case (XMLElement.CDATA, boundary) =>
        /* switch the execution state to CDATA processing so that tokens
         * within the CDATA block are extracted normally (i.e., treated as
         * document content) until the ]]> end-of-section delimiter */
        _state = XMLParserState.CDATA
        delegate.following(boundary - 1)
      case (XMLElement.Comment, _) =>
        delegate.following(XMLBreakIterator.nextEndOfComment(_characters) - 1)
      case (XMLElement.Conditional, _) =>
        delegate.following(XMLBreakIterator.matchedEndOfConditional(_characters) - 1)
      case (XMLElement.Declaration, _) =>
        delegate.following(XMLBreakIterator.nextEndOfTag(_characters) - 1)
      case (XMLElement.DocType, _) =>
        delegate.following(XMLBreakIterator.nextEndOfDTD(_characters) - 1)
      case (XMLElement.ProcessingInstruction, _) =>
        delegate.following(XMLBreakIterator.nextEndOfPI(_characters) - 1)
      case (XMLElement.StartTag, _) =>
        delegate.following(XMLBreakIterator.nextEndOfTag(_characters) - 1)
      case (XMLElement.EndTag, _) =>
        delegate.following(XMLBreakIterator.nextEndOfTag(_characters) - 1)
      }}).getOrElse({
        _lastRuleStatus = 0
        delegate.next
      })
  }

  /** Locates the next boundary within a CDATA section
    *
    * As defined in the XML specification, text within CDATA sections is
    * treated as document content; however, all of the text is treated as
    * just raw characters; there is no markup until the terminating ]]>
    * marker.
    */
  private[this] def nextCDATA: Int = {
    /* check for an end-of-CDATA delimiter sequence. since the end-of-CDATA
     * uses ASCII characters, it's safe to use code unit iteration rather than
     * code point iteration here */
    val isDelim = _characters.next == ']' &&
      _characters.next == ']' &&
      _characters.next == '>'

    if (isDelim) {
      // matched end-of-section marker ]]>, return to normal parsing
      _state = XMLParserState.CharacterData
      _lastRuleStatus = Tag.ruleStatus(Tag.Markup)
      delegate.following(_characters.getIndex - 1)
    } else {
      // still inside CDATA, tokenize the next word
      _lastRuleStatus = 0
      delegate.next
    }
  }

  /** Sets the iterator to analyze a new piece of text.
    *
    * {@link com.ibm.icu.text.BreakIterator#setText}
    * @param text the text to analyze
    */
  override def setText(text: String) {
    super.setText(text)
    _state = XMLParserState.CharacterData
  }

  override def setText(text: java.text.CharacterIterator) {
    super.setText(text)
    _state = XMLParserState.CharacterData
  }

  /** Returns the status tag of the break rule that determined the last boundary
    *
    * {@link com.ibm.icu.text.BreakIterator#getRuleStatus}
    */
  override def getRuleStatus(): Int = {
    if (_lastRuleStatus == 0)
      delegate.getRuleStatus
    else
      _lastRuleStatus
  }
}

object XMLParserState extends Enumeration {
  val CharacterData,
    CDATA = Value
}

object XMLElement extends Enumeration {
  val CDATA,
    Comment,
    Conditional,
    Declaration,
    DocType,
    EndTag,
    EntityDeclaration,
    Notation,
    ProcessingInstruction,
    StartTag = Value
}

private[tokenizer] object XMLBreakIterator {

  val TRIE: Trie[XMLElement.type] = {
    val builder = new CharsTrieBuilder()
    builder.add("<!--", XMLElement.Comment.id)
    builder.add("<![", XMLElement.Conditional.id)
    builder.add("<![CDATA[", XMLElement.CDATA.id)
    builder.add("<!ATTLIST", XMLElement.Declaration.id)
    builder.add("<!DOCTYPE", XMLElement.DocType.id)
    builder.add("<!ELEMENT", XMLElement.Declaration.id)
    builder.add("<!ENTITY", XMLElement.Declaration.id)
    builder.add("<!NOTATION", XMLElement.Declaration.id)
    /* the XML prolog, while technically NOT a processing instruction,
     * can be tokenized perfectly reasonably using the same PI rules */
    builder.add("<?", XMLElement.ProcessingInstruction.id)
    builder.add("</", XMLElement.EndTag.id)
    builder.add("<", XMLElement.StartTag.id)
    new Trie(XMLElement, builder.build(StringTrieBuilder.Option.FAST))
  }

  /** Returns the end of the document type declaration section
    *
    * Detects an optional intSubset section and includes all of the comments
    * and markup contained within it as part of the boundary.
    *
    * @param ch character iterator
    * @return position of the end of the DTD
    */
  def nextEndOfDTD(ch: UCharacterIterator): Int = {
    while (ch.current != UForwardCharacterIterator.DONE &&
        ch.current != '>' && ch.current != '[') {
      if (ch.current == '"' || ch.current == '\'') {
        // swallow the entire literal
        val q = ch.next
        while (ch.current != UForwardCharacterIterator.DONE && ch.next != q) {}
      }
      else ch.next
    }
    ch.current match {
      case '[' =>
        // process all of the markup up to the closing ]
        nextEndOfIntSubset(ch)
        // swallow up to the end-tag
        while (ch.current != UForwardCharacterIterator.DONE && ch.next != '>') {}
        ch.getIndex
      case '>' =>
        // boundary is immediately after the >
        ch.getIndex + 1
      case _ =>
        // DONE; return the length of string
        ch.getIndex
      ch.getIndex + 1
    }
  }

  /** Returns the end of the intSubset section within the DTD
    *
    * The intSubset (https://www.w3.org/TR/xml11/#NT-intSubset) is an optional
    * list of comments, entity references, declarative markup, and processing
    * instructions. le'sigh
    *
    * @param ch character iterator
    * @return boundary position after the closing ']'
    */
  def nextEndOfIntSubset(ch: UCharacterIterator): Int = {
    while (ch.current != UForwardCharacterIterator.DONE && ch.current != ']') {
      if (ch.next == '<') {
        ch.next match {
          case '>' => // skip
          case '?' => nextEndOfPI(ch)
          case '!' => ch.next match {
            case '[' => matchedEndOfConditional(ch)
            case '-' if ch.next == '-' => nextEndOfComment(ch)
            case _ => // malformed XML
          }
          case _ => nextEndOfTag(ch)
        }
      }
    }
    ch.getIndex + (if (ch.current == ']') 1 else 0)
  }

  /** Returns the position of the matching end-of-conditional delimiter
    *
    * Supports nesting of conditional blocks
    */
  def matchedEndOfConditional(ch: UCharacterIterator): Int = {
    /* depth starts at 1, because this is only executed after the outer-
     * most conditional block start is found */
    var depth: Int = 1
    while (ch.current != UForwardCharacterIterator.DONE && depth > 0) {
      ch.next match {
        case '<' if (ch.next == '!' && ch.next == '[') => depth += 1
        case ']' if (ch.next == ']' && ch.next == '>') => depth -= 1
        case _ =>
      }
    }
    ch.getIndex
  }

  /** Returns the position of the next end of comment token
    *
    * @param ch character iterator
    */
  def nextEndOfComment(ch: UCharacterIterator): Int = {
    while (ch.current != UForwardCharacterIterator.DONE &&
      !(ch.next == '-' && ch.next == '-' && ch.next == '>')) {}
    ch.getIndex
  }

  /** Returns the position of the next end of processing instruction marker
    *
    * @param ch character iterator
    */
  def nextEndOfPI(ch: UCharacterIterator): Int = {
    while (ch.current != UForwardCharacterIterator.DONE &&
      !(ch.next == '?' && ch.next == '>')) {}
    ch.getIndex
  }

  /** Returns the position of the next end-of-tag character ('>')
    *
    * Performs some basic parsing to detect literals within the tag, so
    * that a '>' character within the literal does not improperly terminate
    * the tag.
    */
  def nextEndOfTag(ch: UCharacterIterator): Int = {
    while (ch.current != UForwardCharacterIterator.DONE && ch.current != '>') {
      if (ch.current == '"' || ch.current == '\'') {
        // swallow the entire literal
        val q = ch.next
        while (ch.current != UForwardCharacterIterator.DONE && ch.next != q) {}
      }
      else ch.next
    }
    // the boundary is the character after the end-of-tag marker
    ch.getIndex + (if (ch.current == '>') 1 else 0)
  }

  /** Skips over whitespace characters in the iterator
    *
    * @param ch character iterator
    */
  def skipWhitespace(ch: UCharacterIterator) {
    while (isWhitespace(ch.current)) ch.next
  }

  /** Returns true if the character is XML whitespace
    *
    * Ref XML Specification 1.1, 2.3 (https://www.w3.org/TR/xml11/#sec-common-syn)
    *
    * @param ch character
    */
  def isWhitespace(ch: Int) = ch == 0x20 || ch == 0x9 || ch == 0xd || ch == 0xa
}
