package com.idibon.ml.feature.tokenizer

import com.ibm.icu
import org.scalatest.{Matchers, FunSpec}

class XMLBreakIteratorSpec extends FunSpec
    with Matchers with BreakIteratorHelpers[XMLBreakIterator] {

  def newIterator(del: icu.text.BreakIterator) = new XMLBreakIterator(del)

  it("should support empty strings") {
    !!("") shouldBe List("")
  }

  it("should report a negative status tag for custom tokens") {
    tag("<div>text</div>") shouldBe List(
      ("<div>", Tag.ruleStatus(Tag.Markup)), ("text", 200),
      ("</div>", Tag.ruleStatus(Tag.Markup)))
  }

  it("should tokenize comments") {
    !!("hello <!-- comment --> world!") shouldBe List(
      "hello", " ", "<!-- comment -->", " ", "world", "!")
    !!("<!-- comment -->") shouldBe List("<!-- comment -->")
    !!("hello world! <!--com-ment-->") shouldBe List(
      "hello", " ", "world", "!", " ", "<!--com-ment-->")
  }

  it("should tokenize multi-line comments") {
    !!("""<!-- in XML,
comments can extend across multiple lines;
and you just have to deal with them :( --><doc>not a comment</doc>""") shouldBe List(
"""<!-- in XML,
comments can extend across multiple lines;
and you just have to deal with them :( -->""", "<doc>", "not", " ",
  "a", " ", "comment", "</doc>")

  }

  it("should tokenize CDATA") {
    !!("<![CDATA[CDATA can be anything <!-- --> except ]]>") shouldBe List(
      "<![CDATA[", "CDATA", " ", "can", " ", "be", " ",
      "anything", " ", "<", "!", "-", "-", " ", "-", "-", ">",
      " ", "except", " ", "]]>")
    !!("hello<![CDATA[]]>") shouldBe List("hello", "<![CDATA[", "]]>")
    !!("<![CDATA[] ]>]]>") shouldBe List("<![CDATA[", "]", " ", "]", ">", "]]>")
  }

  it("should tokenize processing instructions") {
    !!("""<?xml version="1.1" encoding="UTF-8"?>
<?xml-stylesheet?>""") shouldBe List(
      """<?xml version="1.1" encoding="UTF-8"?>""", "\n", "<?xml-stylesheet?>")
  }

  it("should tokenize basic tags") {
    !!("""<a href="hel<>lo">foo</a>""") shouldBe List(
      """<a href="hel<>lo">""", "foo", "</a>")
    !!("<img src='http://www.>.com' alt='text'/></img>") shouldBe List(
      "<img src='http://www.>.com' alt='text'/>", "</img>")
  }

  it("should tokenize tags defined with unusual names") {
    !!("<!ELEMENT :\ud83d\udca9 (#PCDATA)><:\ud83d\udca9>text</:\ud83d\udca9>") shouldBe List(
      "<!ELEMENT :\ud83d\udca9 (#PCDATA)>", "<:\ud83d\udca9>",
      "text", "</:\ud83d\udca9>")
  }

  it("should tokenize conditional blocks") {
    !!("""<![%foo; [<!ENTITY test 'IGNORE'>]]>hello!
<![%bar; []]>""") shouldBe List(
      "<![%foo; [<!ENTITY test 'IGNORE'>]]>", "hello", "!", "\n", "<![%bar; []]>")
  }

  it("should tokenize declarations") {
    !!("""<!NOTATION gif PUBLIC "gif > viewer">
<!ELEMENT code (#PCDATA)>""") shouldBe List(
      """<!NOTATION gif PUBLIC "gif > viewer">""", "\n", "<!ELEMENT code (#PCDATA)>")
  }

  it("should tokenize simple DTDs") {
    !!("""<!DOCTYPE html PUBLIC "pubid" "system"><doc></doc>""") shouldBe List(
      """<!DOCTYPE html PUBLIC "pubid" "system">""", "<doc>", "</doc>")
  }

  it("should tokenize simple DTDs with brackets in literal strings") {
    !!("""<!DOCTYPE weird "[">done!""") shouldBe List(
      """<!DOCTYPE weird "[">""", "done", "!")
  }

  it("should tokenize complex DTDs") {
    !!("""<!DOCTYPE complex [
<!-- yes, you can put comments <![ in XML DTD sections -->
%this_works;
<!ELEMENT foo>
<![%this-too; [ <![%and-this; [ <!ATTLIST bar> ]]> ]]>
<!-- and more comments :( -->
]
><foo bar="baz">hi</foo>""") shouldBe List(
"""<!DOCTYPE complex [
<!-- yes, you can put comments <![ in XML DTD sections -->
%this_works;
<!ELEMENT foo>
<![%this-too; [ <![%and-this; [ <!ATTLIST bar> ]]> ]]>
<!-- and more comments :( -->
]
>""", """<foo bar="baz">""", "hi", "</foo>")
  }

  it("should support multi-line nested conditional blocks") {
    !!("""<![%foo; [
<![%bar; [
<![%baz; ] ]]>
]]>
]]>foo bar baz""") shouldBe List(
      """<![%foo; [
<![%bar; [
<![%baz; ] ]]>
]]>
]]>""", "foo", " ", "bar", " ", "baz")
  }

  it("should exit if the last tag has no terminator") {
    !!("<![CDATA[Hi!]]") shouldBe List("<![CDATA[", "Hi", "!", "]", "]")
    !!("<!-- com->ment ") shouldBe List("<!-- com->ment ")
    !!("<?xml ? >foo") shouldBe List("<?xml ? >foo")
    !!("<a ") shouldBe List("<a ")
    !!("<![ IGNORE [") shouldBe List("<![ IGNORE [")
    !!("<!-") shouldBe List("<!-")
    !!("<!DOCTYPE [ <!") shouldBe List("<!DOCTYPE [ <!")
  }
}
