package com.idibon.ml.feature.tokenizer

import com.idibon.ml.feature.tokenizer.Unicode._

import com.ibm.icu.lang.UCharacter

/** State machine for parsing RFC-3987 and RFC-2396 compatible URIs
  *
  * This class does not parse the SCHEME part of URIs; clients should
  * parse schemes themselves and then supply an appropriate initial
  * state for the state machine (e.g, AUTHORITY or OPAQUE) based on
  * the expected resource locator for the scheme
  */
class URIStateMachine {

  /** Resets the state machine to a specific state */
  def reset(initial: URIStateMachineState.Value): this.type = {
    _state = initial
    _matched = false
    _hexCount = 0
    this
  }

  /** True if more codepoints may be added to the URI / IRI */
  def hasNext = _state != URIStateMachineState.DONE

  /** True if the state machine has a match for a URI / IRI */
  def hasMatch = _matched

  /** Advances the parser by a codepoint */
  def nextCodePoint(cp: Int): this.type = {
    _state match {
      case URIStateMachineState.AUTHORITY if cp == '/' => {
        // transition to path rules
        _state = URIStateMachineState.PATH
      }
      case URIStateMachineState.PATH |
          URIStateMachineState.AUTHORITY if cp == '?' => {
        // transition to query rules
        _state = URIStateMachineState.QUERY
      }
      case URIStateMachineState.QUERY |
          URIStateMachineState.PATH |
          URIStateMachineState.AUTHORITY if cp == '#' => {
        // transition to fragment rules
        _state = URIStateMachineState.FRAGMENT
      }
      case _ if cp == '%' && _hexCount == 0 => {
        // start of a new percent-encoded pair
        _hexCount = 2
      }
      case _ if _hexCount > 0 && isAsciiHexDigit(cp) => {
        // one less hex digit to match
        _hexCount -= 1
      }
      //==== all state transitions for illegal code points ===
      case _ if _hexCount > 0 => {
        /* invalid percent encoding (non hex-digit where a hex-digit
         * was expected); terminate the state machine */
        _state = URIStateMachineState.DONE
      }
      case URIStateMachineState.OPAQUE if !_matched && cp == '/' => {
        // the first character of an opaque URI may not be a slash
        _state = URIStateMachineState.DONE
      }
      case _ if !URIStateMachine.isValidCharForState(_state, cp) => {
        // break if the codepoint is not allowed in RFC-2396 URIs
        _state = URIStateMachineState.DONE
      }
      case _ => /* consume valid codepoint */
    }

    /* treat any string where at least one full character (possibly percent
     * encoded) matched as a legitimate URI */
    _matched ||= (hasNext && _hexCount == 0)
    this
  }

  // current section of the URI being parsed
  private[this] var _state = URIStateMachineState.DONE
  // track how many percent-encoded characters are expected
  private[this] var _hexCount: Int = 0
  // has the text ever matched a valid URI?
  private[this] var _matched: Boolean = false
}

private[this] object URIStateMachine {
  /** Returns true if the code point is an RFC-2396 unreserved character */
  def isUriUnreservedChar(cp: Int): Boolean = {
    ((cp >= 'a' && cp <= 'z') ||
      (cp >= 'A' && cp <= 'Z') ||
      (cp >= '0' && cp <= '9') ||
      (cp == '-' || cp == '.' || cp == '_' || cp == '~'))
  }

  /** Returns true if the code point is valid in an RFC-2396 opaque URI */
  def isUriOpaqueChar(cp: Int): Boolean = {
    (isUriUnreservedChar(cp) || cp == '/' ||
      cp == ';' || cp == '?' || cp == ':' || cp == '@' ||
      cp == '&' || cp == '=' || cp == '+' || cp == '$' || cp == ',')
  }

  /** Returns true for assigned SMP code points legal in RFC-3987 IRIs */
  def isLegalSMPCodepoint(cp: Int): Boolean = {
    val plane = cp & 0x1f0000
    val index = cp & 0xffff
    (plane != 0 && plane <= 0xe0000 && index <= 0xfffd)
  }

  /** Returns true if a private-use codepoint is legal in an IRI. */
  def isLegalPrivateCodepoint(cp: Int): Boolean = {
    ((cp >= 0xe000 && cp <= 0xf8ff) ||
      (cp >= 0xf0000 && cp <= 0xffffd) ||
      (cp >= 0x100000 && cp <= 0x10fffd))
  }

  /** Returns true if a BMP or SMP codepoint is legal in an IRI */
  def isLegalIriCodepoint(cp: Int): Boolean = {
    ((cp >= 0xa0 && cp <= 0xd7ff) ||
      (cp >= 0xf900 && cp <= 0xfdcf) ||
      (cp >= 0xfdf0 && cp <= 0xffef) ||
      isLegalSMPCodepoint(cp))
  }

  /** Returns true if the code point is a valid RFC-3987 path character */
  def isIriPchar(cp: Int): Boolean = {
    (isUriUnreservedChar(cp) || isLegalIriCodepoint(cp) ||
      cp == '!' || cp == '&' || cp == '\'' || cp == '(' ||
      cp == ')' || cp == '*' || cp == '+' || cp == ',' ||
      cp == ';' || cp == '=' || cp == ':' || cp == '@')
  }

  /** Returns true if the codepoint may be used in an IRI authority component
    *
    * This is a relaxed check, allowing any character allowed by any of the
    * sub-sections in authority (user-info, host, port) anywhere, rather than
    * strictly verifying that the codepoint is valid for the location within
    * the authority.
    */
  def isIriAuthorityChar(cp: Int): Boolean = isIriPathChar(cp)

  /** Returns true if the codepoint may be used in an IRI path component */
  def isIriPathChar(cp: Int): Boolean = (isIriPchar(cp) || cp == '/')

  /** Returns true if the codepoint may be used in an IRI fragment */
  def isIriFragmentChar(cp: Int): Boolean = (isIriPathChar(cp) ||cp == '?')

  /** Returns true if the codepoint may be used in an IRI query component
    *
    * Oddly (and counter-intuitively), this is a super-set of the codepoints
    * allowed for fragments
    */
  def isIriQueryChar(cp: Int): Boolean = {
    (isIriFragmentChar(cp) || isLegalPrivateCodepoint(cp))
  }

  /** Returns true if the codepoint is valid for the specified state */
  def isValidCharForState(s: URIStateMachineState.Value, cp: Int) = s match {
    case URIStateMachineState.AUTHORITY => isIriAuthorityChar(cp)
    case URIStateMachineState.PATH => isIriPathChar(cp)
    case URIStateMachineState.QUERY => isIriQueryChar(cp)
    case URIStateMachineState.FRAGMENT => isIriFragmentChar(cp)
    case URIStateMachineState.OPAQUE => isUriOpaqueChar(cp)
    case _ => false
  }
}

/** States for the state machine correspond to delimited URI components */
private object URIStateMachineState extends Enumeration {
  val AUTHORITY,
    PATH,
    QUERY,
    FRAGMENT,
    OPAQUE,
    DONE = Value
}
