package com.idibon.ml.jruby

import org.json4s._
import scala.collection.JavaConverters._
import org.jruby.runtime.builtin.IRubyObject

/** Empty class to contain statically-callable methods from companion object
  *
  * JRuby doesn't have a good way of accessing the companion class (i.e.,
  * com.idibon.ml.jruby.Json$) directly, so create a stub class so that the
  * Scala compiler creates thunks with static call signatures inside
  * com.idibon.ml.jruby.Json
  */
class Json {
}

object Json {

  /** Converts Ruby-JSON objects into Scala-JSON objects for idiml
    *
    * This method wraps {@link com.idibon.ml.jruby.Json#toIdiml(Object)
    * with a type signature expected by the JRuby runtime (i.e., it
    * accepts an org.jruby.RubyObject parameter).
    *
    * @param ruby - the RubyObject to convert
    * @return the converted Json4s AST
    */
  def toIdiml(ruby: org.jruby.RubyObject): JValue = {
    toIdiml(ruby.asInstanceOf[Object])
  }

  /** Converts Ruby-JSON objects into Scala-JSON objects for idiml
    *
    * Recursively introspects the internal JRuby representation of data
    * objects and converts into a Json4s AST.
    *
    * Converted objects must be JSON-compatible, i.e., all hashes must
    * use either Strings or Symbol as keys, and all values must be Strings,
    * Numbers, Arrays, Hashes, or the Nil value. This method throws
    * an UnsupportedOperationException if the provided object does not meet
    * restrictions.
    *
    * @param ruby - any object from the JRuby environment to convert
    * @return the converted Json4s AST
    */
  def toIdiml(ruby: Object): JValue = ruby match {
    case h: org.jruby.RubyHash => {
      val entries = scala.collection.mutable.ListBuffer[JField]()
      h.visitAll(new org.jruby.RubyHash.Visitor() {
        def visit(key: IRubyObject, value: IRubyObject) {
          val k = toIdiml(key)
          if (!k.isInstanceOf[JString])
            throw new IllegalArgumentException(s"invalid key: $key")
          val v = toIdiml(value)
          entries += JField(k.asInstanceOf[JString].s, v)
        }
      })
      JObject(entries.toList)
    }
    case a: org.jruby.RubyArray => {
      JArray(a.asScala.map(i => toIdiml(i.asInstanceOf[Object])).toList)
    }
    /* the JRuby runtime keeps some Java primitive types (Strings,
     * Numbers) internally in compound objects, so we need to detect
     * and convert those, too. */
    case s: org.jruby.RubyString => JString(s.decodeString)
    case s: org.jruby.RubySymbol => JString(s.asJavaString)
    case s: String => JString(s)
    case i: org.jruby.RubyInteger => JInt(i.getLongValue)
    case i: java.lang.Long => JInt(i.longValue)
    case n: org.jruby.RubyNumeric => JDouble(n.getDoubleValue)
    case n: java.lang.Number => JDouble(n.doubleValue)
    case _: org.jruby.RubyBoolean.True => JBool(true)
    case _: org.jruby.RubyBoolean.False => JBool(false)
    case b: java.lang.Boolean => JBool(b.booleanValue)
    case _: org.jruby.RubyNil => JNothing
    case null => JNothing

    case _ => throw new UnsupportedOperationException(s"${ruby.getClass}")
  }
}
