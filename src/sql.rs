//! 最小 SQL WHERE 子集解析器与谓词求值。
//!
//! 支持的语义：`=`, `!=`, `<>`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `IN`。
//! 解析结果是一颗表达式树，可在每条向量的 payload 上独立求值。
//!
//! 分区级下推：
//! 对每个分区维护数值字段的 min/max 与字符串字段的小集合；
//! 若谓词与分区统计绝对矛盾，则跳过整个分区，避免读取无关码。
//! 下推是安全的：返回 false 表示“该分区不可能有满足条件的向量”，
//! 返回 true 表示“不确定，需进入分区逐向量求值”。

use serde_json::Value;
use std::fmt;

use crate::index_ivf_rq::{PartitionStats, Payload};

#[cfg(test)]
use serde_json::json;

/// SQL 谓词错误。
#[derive(Debug)]
pub struct SqlError(String);

impl fmt::Display for SqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SQL parse error: {}", self.0)
    }
}

impl std::error::Error for SqlError {}

/// 谓词节点。
#[derive(Debug, Clone)]
pub enum SqlPredicate {
    Eq(String, ScalarValue),
    Ne(String, ScalarValue),
    Lt(String, ScalarValue),
    Le(String, ScalarValue),
    Gt(String, ScalarValue),
    Ge(String, ScalarValue),
    In(String, Vec<ScalarValue>),
    And(Box<SqlPredicate>, Box<SqlPredicate>),
    Or(Box<SqlPredicate>, Box<SqlPredicate>),
}

/// 外部可复用的标量值类型。
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Number(f64),
    String(String),
    Bool(bool),
}

impl SqlPredicate {
    /// 在 payload 上求值。
    pub fn eval(&self, payload: &Payload) -> bool {
        match self {
            SqlPredicate::Eq(field, value) => compare(payload, field, value, OrderingPred::Eq),
            SqlPredicate::Ne(field, value) => !compare(payload, field, value, OrderingPred::Eq),
            SqlPredicate::Lt(field, value) => compare(payload, field, value, OrderingPred::Lt),
            SqlPredicate::Le(field, value) => {
                compare(payload, field, value, OrderingPred::Lt)
                    || compare(payload, field, value, OrderingPred::Eq)
            }
            SqlPredicate::Gt(field, value) => compare(payload, field, value, OrderingPred::Gt),
            SqlPredicate::Ge(field, value) => {
                compare(payload, field, value, OrderingPred::Gt)
                    || compare(payload, field, value, OrderingPred::Eq)
            }
            SqlPredicate::In(field, values) => {
                let payload_value = payload.get(field);
                values.iter().any(|v| values_equal(payload_value, v))
            }
            SqlPredicate::And(a, b) => a.eval(payload) && b.eval(payload),
            SqlPredicate::Or(a, b) => a.eval(payload) || b.eval(payload),
        }
    }
}

#[derive(Clone, Copy)]
enum OrderingPred {
    Lt,
    Eq,
    Gt,
}

fn compare(payload: &Payload, field: &str, value: &ScalarValue, pred: OrderingPred) -> bool {
    let payload_value = payload.get(field);
    let ord = value_partial_cmp(payload_value, value);
    match pred {
        OrderingPred::Lt => ord == Some(std::cmp::Ordering::Less),
        OrderingPred::Eq => ord == Some(std::cmp::Ordering::Equal),
        OrderingPred::Gt => ord == Some(std::cmp::Ordering::Greater),
    }
}

fn value_partial_cmp(
    payload_value: Option<&Value>,
    value: &ScalarValue,
) -> Option<std::cmp::Ordering> {
    match (payload_value, value) {
        (Some(Value::Number(n)), ScalarValue::Number(v)) => {
            n.as_f64().and_then(|a| a.partial_cmp(v))
        }
        (Some(Value::String(s)), ScalarValue::String(v)) => Some(s.cmp(v)),
        (Some(Value::Bool(b)), ScalarValue::Bool(v)) => Some(b.cmp(v)),
        _ => None,
    }
}

fn values_equal(payload_value: Option<&Value>, value: &ScalarValue) -> bool {
    match (payload_value, value) {
        (Some(Value::Number(n)), ScalarValue::Number(v)) => n.as_f64() == Some(*v),
        (Some(Value::String(s)), ScalarValue::String(v)) => s == v,
        (Some(Value::Bool(b)), ScalarValue::Bool(v)) => b == v,
        (None, _) => false,
        _ => false,
    }
}

/// 判断分区是否可能包含满足谓词的向量。
///
/// 返回 false：该分区绝对不满足，可跳过。
/// 返回 true：不确定，需要进入分区逐向量求值。
/// 注意：当前实现只下推数值字段的 min/max 和字符串字段的小集合；
/// 不支持的谓词保守返回 true。
pub fn can_partition_match(predicate: &SqlPredicate, stats: &PartitionStats) -> bool {
    match predicate {
        SqlPredicate::Eq(field, ScalarValue::Number(v)) => {
            if let Some((min, max)) = stats.num_ranges.get(field) {
                // 若目标值不在分区数值范围内，则不可能相等。
                *v >= *min && *v <= *max
            } else {
                true
            }
        }
        SqlPredicate::Eq(field, ScalarValue::String(v)) => {
            if let Some(set) = stats.string_values.get(field) {
                // 若目标字符串不在分区取值集合中，则不可能相等。
                set.contains(v)
            } else {
                true
            }
        }
        SqlPredicate::Ne(_, _) => true,
        SqlPredicate::Lt(field, ScalarValue::Number(v)) => {
            if let Some((min, _)) = stats.num_ranges.get(field) {
                // 分区最小值已经 >= v，则不可能有 < v 的向量。
                *min < *v
            } else {
                true
            }
        }
        SqlPredicate::Le(field, ScalarValue::Number(v)) => {
            if let Some((min, _)) = stats.num_ranges.get(field) {
                *min <= *v
            } else {
                true
            }
        }
        SqlPredicate::Gt(field, ScalarValue::Number(v)) => {
            if let Some((_, max)) = stats.num_ranges.get(field) {
                // 分区最大值已经 <= v，则不可能有 > v 的向量。
                *max > *v
            } else {
                true
            }
        }
        SqlPredicate::Ge(field, ScalarValue::Number(v)) => {
            if let Some((_, max)) = stats.num_ranges.get(field) {
                *max >= *v
            } else {
                true
            }
        }
        SqlPredicate::In(field, values) => {
            // 若分区取值集合已知，且没有任何 IN 值在其中，则可跳过。
            if let Some(set) = stats.string_values.get(field) {
                values.iter().any(|v| {
                    if let ScalarValue::String(s) = v {
                        set.contains(s)
                    } else {
                        false
                    }
                })
            } else if stats.num_ranges.contains_key(field) {
                // 数值字段：只要有一个值落在 [min, max] 内就保留。
                if let Some((min, max)) = stats.num_ranges.get(field) {
                    values.iter().any(|v| {
                        if let ScalarValue::Number(n) = v {
                            *n >= *min && *n <= *max
                        } else {
                            false
                        }
                    })
                } else {
                    true
                }
            } else {
                true
            }
        }
        SqlPredicate::And(a, b) => {
            // AND：任一子谓词判定不可满足则整个分区跳过。
            can_partition_match(a, stats) && can_partition_match(b, stats)
        }
        SqlPredicate::Or(a, b) => {
            // OR：两个子谓词都不可满足才跳过。
            can_partition_match(a, stats) || can_partition_match(b, stats)
        }
        _ => true,
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Number(f64),
    String(String),
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LParen,
    RParen,
    Comma,
    And,
    Or,
    In,
}

fn tokenize(input: &str) -> Result<Vec<Token>, SqlError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != quote {
                s.push(chars[i]);
                i += 1;
            }
            if i >= chars.len() {
                return Err(SqlError("unterminated string".to_string()));
            }
            i += 1; // skip closing quote
            tokens.push(Token::String(s));
            continue;
        }
        if c.is_ascii_digit() || (c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let mut s = String::new();
            if c == '-' {
                s.push(c);
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                s.push(chars[i]);
                i += 1;
            }
            let v: f64 = s
                .parse()
                .map_err(|_| SqlError(format!("bad number: {}", s)))?;
            tokens.push(Token::Number(v));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let mut s = String::new();
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
            {
                s.push(chars[i]);
                i += 1;
            }
            let upper = s.to_ascii_uppercase();
            tokens.push(match upper.as_str() {
                "AND" => Token::And,
                "OR" => Token::Or,
                "IN" => Token::In,
                "TRUE" => Token::Number(1.0),
                "FALSE" => Token::Number(0.0),
                _ => Token::Ident(s),
            });
            continue;
        }
        // operators
        let two = if i + 1 < chars.len() {
            let mut s = String::new();
            s.push(c);
            s.push(chars[i + 1]);
            s
        } else {
            String::new()
        };
        match two.as_str() {
            "!=" | "<>" => {
                tokens.push(Token::Ne);
                i += 2;
                continue;
            }
            "<=" => {
                tokens.push(Token::Le);
                i += 2;
                continue;
            }
            ">=" => {
                tokens.push(Token::Ge);
                i += 2;
                continue;
            }
            _ => {}
        }
        match c {
            '=' => tokens.push(Token::Eq),
            '<' => tokens.push(Token::Lt),
            '>' => tokens.push(Token::Gt),
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            ',' => tokens.push(Token::Comma),
            _ => return Err(SqlError(format!("unexpected char: {}", c))),
        }
        i += 1;
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn parse(mut self) -> Result<SqlPredicate, SqlError> {
        let expr = self.parse_or()?;
        if self.pos != self.tokens.len() {
            return Err(SqlError("trailing tokens".to_string()));
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<SqlPredicate, SqlError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = SqlPredicate::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<SqlPredicate, SqlError> {
        let mut left = self.parse_comparison()?;
        while self.peek() == Some(&Token::And) {
            self.advance();
            let right = self.parse_comparison()?;
            left = SqlPredicate::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<SqlPredicate, SqlError> {
        if self.peek() == Some(&Token::LParen) {
            self.advance();
            let inner = self.parse_or()?;
            self.expect(&Token::RParen)?;
            return Ok(inner);
        }
        let field = self.expect_ident()?;
        if self.peek() == Some(&Token::In) {
            self.advance();
            self.expect(&Token::LParen)?;
            let mut values = Vec::new();
            values.push(self.expect_scalar()?);
            while self.peek() == Some(&Token::Comma) {
                self.advance();
                values.push(self.expect_scalar()?);
            }
            self.expect(&Token::RParen)?;
            return Ok(SqlPredicate::In(field, values));
        }
        let op = self
            .next()
            .cloned()
            .ok_or_else(|| SqlError("expected operator".to_string()))?;
        let value = self.expect_scalar()?;
        match op {
            Token::Eq => Ok(SqlPredicate::Eq(field, value)),
            Token::Ne => Ok(SqlPredicate::Ne(field, value)),
            Token::Lt => Ok(SqlPredicate::Lt(field, value)),
            Token::Le => Ok(SqlPredicate::Le(field, value)),
            Token::Gt => Ok(SqlPredicate::Gt(field, value)),
            Token::Ge => Ok(SqlPredicate::Ge(field, value)),
            _ => Err(SqlError("expected comparison operator".to_string())),
        }
    }

    fn expect_scalar(&mut self) -> Result<ScalarValue, SqlError> {
        match self.next().cloned() {
            Some(Token::Number(v)) => Ok(ScalarValue::Number(v)),
            Some(Token::String(s)) => Ok(ScalarValue::String(s)),
            _ => Err(SqlError("expected scalar value".to_string())),
        }
    }

    fn expect_ident(&mut self) -> Result<String, SqlError> {
        match self.next().cloned() {
            Some(Token::Ident(s)) => Ok(s),
            _ => Err(SqlError("expected identifier".to_string())),
        }
    }

    fn expect(&mut self, token: &Token) -> Result<(), SqlError> {
        match self.next() {
            Some(t) if t == token => Ok(()),
            _ => Err(SqlError(format!("expected {:?}", token))),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn next(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
}

/// 将 SQL WHERE 子集字符串解析为谓词树。
pub fn parse_sql_filter(input: &str) -> Result<SqlPredicate, SqlError> {
    let tokens = tokenize(input)?;
    Parser { tokens, pos: 0 }.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(field: &str, value: Value) -> Payload {
        let mut m = Payload::new();
        m.insert(field.to_string(), value);
        m
    }

    #[test]
    fn test_eq_and_numeric_compare() {
        let pred = parse_sql_filter("age >= 18 AND name = 'alice'").unwrap();
        let p1 = payload("age", json!(20));
        let p1 = {
            let mut m = p1;
            m.insert("name".to_string(), json!("alice"));
            m
        };
        assert!(pred.eval(&p1));
        let p2 = payload("age", json!(16));
        assert!(!pred.eval(&p2));
    }

    #[test]
    fn test_in_or() {
        let pred = parse_sql_filter("category IN ('a', 'b') OR score > 90").unwrap();
        let p = payload("category", json!("a"));
        assert!(pred.eval(&p));
        let p = payload("score", json!(95));
        assert!(pred.eval(&p));
        let p = payload("score", json!(80));
        assert!(!pred.eval(&p));
    }
}
