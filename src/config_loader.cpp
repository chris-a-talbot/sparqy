#include "config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct SourceLocation {
    size_t line = 1u;
    size_t column = 1u;
};

enum class TokenType : uint8_t {
    identifier,
    number,
    string_literal,
    lparen,
    rparen,
    comma,
    equal,
    arrow,
    end_of_file
};

struct Token {
    TokenType type = TokenType::end_of_file;
    std::string text;
    SourceLocation loc;
};

[[noreturn]] void throw_config_error(const std::string& path,
                                     const SourceLocation& loc,
                                     const std::string& message) {
    std::ostringstream oss;
    oss << path << ":" << loc.line << ":" << loc.column << ": " << message;
    throw std::runtime_error(oss.str());
}

std::string format_warning(const std::string& path,
                           const SourceLocation& loc,
                           const std::string& message) {
    std::ostringstream oss;
    oss << path << ":" << loc.line << ":" << loc.column << ": " << message;
    return oss.str();
}

std::string read_file_contents(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error("unable to open config file '" + path + "'");
    }

    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("unable to determine size of config file '" + path + "'");
    }
    in.seekg(0, std::ios::beg);

    std::string contents((size_t)size, '\0');
    if (size > 0 && !in.read(contents.data(), size)) {
        throw std::runtime_error("unable to read config file '" + path + "'");
    }
    return contents;
}

bool is_identifier_start(char c) {
    const unsigned char uc = (unsigned char)c;
    return std::isalpha(uc) || c == '_' || c == '.';
}

bool is_identifier_body(char c) {
    const unsigned char uc = (unsigned char)c;
    return std::isalnum(uc) || c == '_' || c == '.';
}

bool is_number_start(const std::string& text, size_t index) {
    const char c = text[index];
    if (std::isdigit((unsigned char)c)) return true;
    if (c == '.' && index + 1u < text.size()
        && std::isdigit((unsigned char)text[index + 1u])) {
        return true;
    }
    if ((c == '+' || c == '-')
        && index + 1u < text.size()
        && (std::isdigit((unsigned char)text[index + 1u]) || text[index + 1u] == '.')) {
        return true;
    }
    return false;
}

class Tokenizer {
public:
    Tokenizer(std::string path, std::string source)
        : path_(std::move(path)), source_(std::move(source)) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        tokens.reserve(std::max<size_t>(16u, source_.size() / 4u));
        while (!at_end()) {
            skip_ignored();
            if (at_end()) break;

            const SourceLocation loc = location();
            const char c = peek();
            if (is_identifier_start(c)) {
                tokens.push_back(tokenize_identifier(loc));
                continue;
            }
            if (is_number_start(source_, index_)) {
                tokens.push_back(tokenize_number(loc));
                continue;
            }
            if (c == '"' || c == '\'') {
                tokens.push_back(tokenize_string(loc));
                continue;
            }

            switch (c) {
                case '(':
                    tokens.push_back(simple_token(TokenType::lparen, "(", loc));
                    advance();
                    break;
                case ')':
                    tokens.push_back(simple_token(TokenType::rparen, ")", loc));
                    advance();
                    break;
                case ',':
                    tokens.push_back(simple_token(TokenType::comma, ",", loc));
                    advance();
                    break;
                case '=':
                    tokens.push_back(simple_token(TokenType::equal, "=", loc));
                    advance();
                    break;
                case '<':
                    if (peek_next() == '-') {
                        tokens.push_back(simple_token(TokenType::arrow, "<-", loc));
                        advance();
                        advance();
                        break;
                    }
                    throw_config_error(path_, loc, "unexpected '<'; expected '<-'");
                default: {
                    std::ostringstream oss;
                    oss << "unexpected character '" << c << "'";
                    throw_config_error(path_, loc, oss.str());
                }
            }
        }

        tokens.push_back(Token{TokenType::end_of_file, "", location()});
        return tokens;
    }

private:
    bool at_end() const { return index_ >= source_.size(); }

    char peek() const { return source_[index_]; }

    char peek_next() const {
        return (index_ + 1u < source_.size()) ? source_[index_ + 1u] : '\0';
    }

    void advance() {
        if (at_end()) return;
        if (source_[index_] == '\n') {
            ++line_;
            column_ = 1u;
        } else {
            ++column_;
        }
        ++index_;
    }

    SourceLocation location() const { return SourceLocation{line_, column_}; }

    void skip_ignored() {
        while (!at_end()) {
            const char c = peek();
            if (c == '#') {
                while (!at_end() && peek() != '\n') advance();
                continue;
            }
            if (std::isspace((unsigned char)c)) {
                advance();
                continue;
            }
            break;
        }
    }

    Token simple_token(TokenType type,
                       const char* text,
                       const SourceLocation& loc) const {
        return Token{type, text, loc};
    }

    Token tokenize_identifier(const SourceLocation& loc) {
        const size_t begin = index_;
        while (!at_end() && is_identifier_body(peek())) {
            advance();
        }
        return Token{
            TokenType::identifier,
            source_.substr(begin, index_ - begin),
            loc
        };
    }

    Token tokenize_number(const SourceLocation& loc) {
        const size_t begin = index_;
        if ((peek() == '+' || peek() == '-')
            && (std::isdigit((unsigned char)peek_next()) || peek_next() == '.')) {
            advance();
        }

        bool saw_dot = false;
        while (!at_end()) {
            const char c = peek();
            if (std::isdigit((unsigned char)c)) {
                advance();
                continue;
            }
            if (c == '.' && !saw_dot) {
                saw_dot = true;
                advance();
                continue;
            }
            break;
        }

        if (!at_end() && (peek() == 'e' || peek() == 'E')) {
            advance();
            if (!at_end() && (peek() == '+' || peek() == '-')) {
                advance();
            }
            if (at_end() || !std::isdigit((unsigned char)peek())) {
                throw_config_error(path_, location(), "invalid exponent in numeric literal");
            }
            while (!at_end() && std::isdigit((unsigned char)peek())) {
                advance();
            }
        }

        return Token{
            TokenType::number,
            source_.substr(begin, index_ - begin),
            loc
        };
    }

    Token tokenize_string(const SourceLocation& loc) {
        const char quote = peek();
        advance();

        std::string text;
        while (!at_end()) {
            const char c = peek();
            if (c == quote) {
                advance();
                return Token{TokenType::string_literal, std::move(text), loc};
            }
            if (c == '\\') {
                advance();
                if (at_end())
                    throw_config_error(path_, location(), "unterminated escape sequence in string literal");
                const char escaped = peek();
                switch (escaped) {
                    case '\\': text.push_back('\\'); break;
                    case '\'': text.push_back('\''); break;
                    case '"':  text.push_back('"'); break;
                    case 'n':  text.push_back('\n'); break;
                    case 't':  text.push_back('\t'); break;
                    case 'r':  text.push_back('\r'); break;
                    default: {
                        std::ostringstream oss;
                        oss << "unsupported escape sequence '\\" << escaped << "'";
                        throw_config_error(path_, location(), oss.str());
                    }
                }
                advance();
                continue;
            }
            if (c == '\n')
                throw_config_error(path_, loc, "unterminated string literal");
            text.push_back(c);
            advance();
        }

        throw_config_error(path_, loc, "unterminated string literal");
    }

    std::string path_;
    std::string source_;
    size_t index_ = 0u;
    size_t line_ = 1u;
    size_t column_ = 1u;
};

struct Expr;

struct CallArg {
    bool has_name = false;
    std::string name;
    std::unique_ptr<Expr> value;
};

struct Expr {
    enum class Kind : uint8_t {
        integer_number,
        floating_number,
        boolean,
        string_literal,
        symbol,
        call
    };

    Kind kind = Kind::symbol;
    SourceLocation loc;
    std::string text;
    bool bool_value = false;
    std::vector<CallArg> args;
};

struct Assignment {
    std::string name;
    SourceLocation loc;
    std::unique_ptr<Expr> value;
};

class Parser {
public:
    Parser(std::string path, std::vector<Token> tokens)
        : path_(std::move(path)), tokens_(std::move(tokens)) {}

    std::vector<Assignment> parse_program() {
        std::vector<Assignment> assignments;
        assignments.reserve(3u);
        while (!check(TokenType::end_of_file)) {
            assignments.push_back(parse_assignment());
        }
        return assignments;
    }

private:
    Assignment parse_assignment() {
        const Token& name_token = consume(TokenType::identifier, "expected a top-level section name");
        if (!match(TokenType::arrow) && !match(TokenType::equal)) {
            throw_config_error(
                path_,
                peek().loc,
                "expected '<-' or '=' after top-level section name");
        }

        Assignment assignment;
        assignment.name = name_token.text;
        assignment.loc = name_token.loc;
        assignment.value = parse_expr();
        return assignment;
    }

    std::unique_ptr<Expr> parse_expr() {
        return parse_primary();
    }

    std::unique_ptr<Expr> parse_primary() {
        if (check(TokenType::number)) {
            const Token token = advance();
            auto expr = std::make_unique<Expr>();
            expr->loc = token.loc;
            expr->text = token.text;
            expr->kind = (token.text.find_first_of(".eE") == std::string::npos)
                             ? Expr::Kind::integer_number
                             : Expr::Kind::floating_number;
            return expr;
        }

        if (check(TokenType::string_literal)) {
            const Token token = advance();
            auto expr = std::make_unique<Expr>();
            expr->kind = Expr::Kind::string_literal;
            expr->loc = token.loc;
            expr->text = token.text;
            return expr;
        }

        if (check(TokenType::identifier)) {
            const Token token = advance();
            if (match(TokenType::lparen)) {
                auto expr = std::make_unique<Expr>();
                expr->kind = Expr::Kind::call;
                expr->loc = token.loc;
                expr->text = token.text;
                if (!check(TokenType::rparen)) {
                    for (;;) {
                        CallArg arg;
                        if (check(TokenType::identifier) && peek_next().type == TokenType::equal) {
                            arg.has_name = true;
                            arg.name = advance().text;
                            consume(TokenType::equal, "expected '=' after named argument");
                        }
                        arg.value = parse_expr();
                        expr->args.push_back(std::move(arg));
                        if (match(TokenType::comma)) continue;
                        break;
                    }
                }
                consume(TokenType::rparen, "expected ')' to close call");
                return expr;
            }

            auto expr = std::make_unique<Expr>();
            expr->loc = token.loc;
            if (token.text == "TRUE" || token.text == "FALSE") {
                expr->kind = Expr::Kind::boolean;
                expr->bool_value = (token.text == "TRUE");
            } else {
                expr->kind = Expr::Kind::symbol;
                expr->text = token.text;
            }
            return expr;
        }

        throw_config_error(path_, peek().loc, "expected a value");
    }

    bool check(TokenType type) const {
        return peek().type == type;
    }

    bool match(TokenType type) {
        if (!check(type)) return false;
        advance();
        return true;
    }

    Token consume(TokenType type, const char* message) {
        if (!check(type)) throw_config_error(path_, peek().loc, message);
        return advance();
    }

    const Token& peek() const { return tokens_[cursor_]; }

    const Token& peek_next() const {
        const size_t next = std::min(cursor_ + 1u, tokens_.size() - 1u);
        return tokens_[next];
    }

    Token advance() { return tokens_[cursor_++]; }

    std::string path_;
    std::vector<Token> tokens_;
    size_t cursor_ = 0u;
};

using ScalarValue = std::variant<int64_t, double, bool, std::string>;

struct ConstantEntry {
    ScalarValue value;
    SourceLocation loc;
};

struct ConstantScope {
    std::unordered_map<std::string, ConstantEntry> values;
    std::unordered_set<std::string> used_names;
};

struct PendingStatisticSpec {
    StatisticKind kind = StatisticKind::mean_fitness;
    HaplotypeSimilarityMetric metric = HaplotypeSimilarityMetric::jaccard;
    bool every_generation = false;
    std::vector<uint64_t> generations;
    SourceLocation loc;
};

struct SplitArgs {
    std::vector<const Expr*> positional;
    std::unordered_map<std::string, const Expr*> named;
};

SplitArgs split_args(const std::string& path, const Expr& call) {
    SplitArgs split;
    split.positional.reserve(call.args.size());
    split.named.reserve(call.args.size());
    for (const CallArg& arg : call.args) {
        if (arg.has_name) {
            if (split.named.count(arg.name) != 0u) {
                throw_config_error(
                    path,
                    arg.value->loc,
                    "duplicate named argument '" + arg.name + "'");
            }
            split.named[arg.name] = arg.value.get();
        } else {
            split.positional.push_back(arg.value.get());
        }
    }
    return split;
}

class ArgReader {
public:
    ArgReader(const std::string& path, const Expr& call)
        : path_(path),
          call_(call),
          split_(split_args(path, call)),
          used_positional_(split_.positional.size(), false) {}

    const Expr* optional(const std::string& name) {
        const auto named_it = split_.named.find(name);
        if (named_it != split_.named.end()) {
            used_named_.insert(name);
            return named_it->second;
        }
        return take_next_unused_positional();
    }

    const Expr& required(const std::string& name,
                         const std::string& context) {
        const Expr* value = optional(name);
        if (value == nullptr) {
            throw_config_error(
                path_,
                call_.loc,
                context + " requires argument '" + name + "'");
        }
        return *value;
    }

    void require_no_extra(const std::string& context) const {
        for (size_t i = 0; i < used_positional_.size(); ++i) {
            if (!used_positional_[i]) {
                throw_config_error(
                    path_,
                    split_.positional[i]->loc,
                    context + " received an unexpected positional argument");
            }
        }
        for (const auto& named : split_.named) {
            if (used_named_.count(named.first) == 0u) {
                throw_config_error(
                    path_,
                    named.second->loc,
                    context + " received an unexpected named argument '" + named.first + "'");
            }
        }
    }

    size_t positional_count() const { return split_.positional.size(); }

    const std::vector<const Expr*>& positional() const { return split_.positional; }

    std::vector<const Expr*> take_remaining_positional() {
        std::vector<const Expr*> unused;
        unused.reserve(split_.positional.size());
        for (size_t i = 0u; i < split_.positional.size(); ++i) {
            if (!used_positional_[i]) {
                unused.push_back(split_.positional[i]);
                used_positional_[i] = true;
            }
        }
        return unused;
    }

private:
    const Expr* take_next_unused_positional() {
        while (next_positional_index_ < split_.positional.size()
               && used_positional_[next_positional_index_]) {
            ++next_positional_index_;
        }
        if (next_positional_index_ >= split_.positional.size()) {
            return nullptr;
        }
        used_positional_[next_positional_index_] = true;
        return split_.positional[next_positional_index_++];
    }

    const std::string& path_;
    const Expr& call_;
    SplitArgs split_;
    mutable std::vector<bool> used_positional_;
    mutable std::unordered_set<std::string> used_named_;
    size_t next_positional_index_ = 0u;
};

const Expr& expect_call_named(const std::string& path,
                              const Expr& expr,
                              const std::string& call_name,
                              const std::string& context) {
    if (expr.kind != Expr::Kind::call || expr.text != call_name) {
        throw_config_error(
            path,
            expr.loc,
            "expected " + context + " to be " + call_name + "(...)");
    }
    return expr;
}

const Expr& expect_list_call(const std::string& path,
                             const Expr& expr,
                             const std::string& context) {
    return expect_call_named(path, expr, "list", context);
}

int64_t parse_int64_literal(const std::string& path, const Expr& expr) {
    try {
        size_t consumed = 0u;
        const long long value = std::stoll(expr.text, &consumed, 10);
        if (consumed != expr.text.size())
            throw_config_error(path, expr.loc, "invalid integer literal '" + expr.text + "'");
        return (int64_t)value;
    } catch (const std::exception&) {
        throw_config_error(path, expr.loc, "invalid integer literal '" + expr.text + "'");
    }
}

double parse_double_literal(const std::string& path, const Expr& expr) {
    try {
        size_t consumed = 0u;
        const double value = std::stod(expr.text, &consumed);
        if (consumed != expr.text.size())
            throw_config_error(path, expr.loc, "invalid numeric literal '" + expr.text + "'");
        return value;
    } catch (const std::exception&) {
        throw_config_error(path, expr.loc, "invalid numeric literal '" + expr.text + "'");
    }
}

bool try_get_constant(const ConstantScope& constants,
                      const std::string& name,
                      const ConstantEntry*& out_entry) {
    const auto it = constants.values.find(name);
    if (it == constants.values.end()) return false;
    out_entry = &it->second;
    return true;
}

ScalarValue evaluate_constant_literal(const std::string& path, const Expr& expr) {
    switch (expr.kind) {
        case Expr::Kind::integer_number:
            return parse_int64_literal(path, expr);
        case Expr::Kind::floating_number:
            return parse_double_literal(path, expr);
        case Expr::Kind::boolean:
            return expr.bool_value;
        case Expr::Kind::string_literal:
            return expr.text;
        default:
            throw_config_error(
                path,
                expr.loc,
                "constants may only contain scalar literals (integer, float, bool, or string)");
    }
}

double scalar_to_double(const std::string& path,
                        const SourceLocation& loc,
                        const ScalarValue& value,
                        const std::string& context) {
    if (const auto* i = std::get_if<int64_t>(&value)) return (double)*i;
    if (const auto* d = std::get_if<double>(&value)) return *d;
    throw_config_error(path, loc, context + " must be numeric");
}

int64_t scalar_to_int64(const std::string& path,
                        const SourceLocation& loc,
                        const ScalarValue& value,
                        const std::string& context) {
    if (const auto* i = std::get_if<int64_t>(&value)) return *i;
    if (const auto* d = std::get_if<double>(&value)) {
        const double integral = std::llround(*d);
        if (std::abs(*d - integral) > 1e-12)
            throw_config_error(path, loc, context + " must be an integer");
        return (int64_t)integral;
    }
    throw_config_error(path, loc, context + " must be numeric");
}

double resolve_double(const std::string& path,
                      const Expr& expr,
                      ConstantScope& constants,
                      const std::string& context) {
    switch (expr.kind) {
        case Expr::Kind::integer_number:
        case Expr::Kind::floating_number:
            return parse_double_literal(path, expr);
        case Expr::Kind::symbol: {
            const ConstantEntry* entry = nullptr;
            if (!try_get_constant(constants, expr.text, entry)) {
                throw_config_error(path, expr.loc, "unknown constant '" + expr.text + "'");
            }
            constants.used_names.insert(expr.text);
            return scalar_to_double(path, expr.loc, entry->value, context);
        }
        default:
            throw_config_error(path, expr.loc, context + " must be numeric");
    }
}

int resolve_int(const std::string& path,
                const Expr& expr,
                ConstantScope& constants,
                const std::string& context) {
    int64_t value = 0;
    switch (expr.kind) {
        case Expr::Kind::integer_number:
        case Expr::Kind::floating_number:
            value = scalar_to_int64(path,
                                    expr.loc,
                                    (expr.kind == Expr::Kind::integer_number)
                                        ? ScalarValue(parse_int64_literal(path, expr))
                                        : ScalarValue(parse_double_literal(path, expr)),
                                    context);
            break;
        case Expr::Kind::symbol: {
            const ConstantEntry* entry = nullptr;
            if (!try_get_constant(constants, expr.text, entry)) {
                throw_config_error(path, expr.loc, "unknown constant '" + expr.text + "'");
            }
            constants.used_names.insert(expr.text);
            value = scalar_to_int64(path, expr.loc, entry->value, context);
            break;
        }
        default:
            throw_config_error(path, expr.loc, context + " must be an integer");
    }

    if (value < (int64_t)std::numeric_limits<int>::min()
        || value > (int64_t)std::numeric_limits<int>::max()) {
        throw_config_error(path, expr.loc, context + " is out of range for int");
    }
    return (int)value;
}

uint64_t resolve_uint64(const std::string& path,
                        const Expr& expr,
                        ConstantScope& constants,
                        const std::string& context) {
    int64_t signed_value = 0;
    switch (expr.kind) {
        case Expr::Kind::integer_number:
            signed_value = parse_int64_literal(path, expr);
            break;
        case Expr::Kind::floating_number:
            signed_value = scalar_to_int64(
                path, expr.loc, ScalarValue(parse_double_literal(path, expr)), context);
            break;
        case Expr::Kind::symbol: {
            const ConstantEntry* entry = nullptr;
            if (!try_get_constant(constants, expr.text, entry)) {
                throw_config_error(path, expr.loc, "unknown constant '" + expr.text + "'");
            }
            constants.used_names.insert(expr.text);
            signed_value = scalar_to_int64(path, expr.loc, entry->value, context);
            break;
        }
        default:
            throw_config_error(path, expr.loc, context + " must be a non-negative integer");
    }

    if (signed_value < 0)
        throw_config_error(path, expr.loc, context + " must be non-negative");
    return (uint64_t)signed_value;
}

bool resolve_bool(const std::string& path,
                  const Expr& expr,
                  ConstantScope& constants,
                  const std::string& context) {
    if (expr.kind == Expr::Kind::boolean) return expr.bool_value;
    if (expr.kind == Expr::Kind::symbol) {
        const ConstantEntry* entry = nullptr;
        if (!try_get_constant(constants, expr.text, entry)) {
            throw_config_error(path, expr.loc, "unknown constant '" + expr.text + "'");
        }
        constants.used_names.insert(expr.text);
        if (const auto* value = std::get_if<bool>(&entry->value)) return *value;
    }
    throw_config_error(path, expr.loc, context + " must be TRUE or FALSE");
}

std::string resolve_name(const std::string& path,
                         const Expr& expr,
                         ConstantScope& constants,
                         const std::string& context) {
    if (expr.kind == Expr::Kind::string_literal) return expr.text;
    if (expr.kind == Expr::Kind::symbol) {
        const ConstantEntry* entry = nullptr;
        if (try_get_constant(constants, expr.text, entry)) {
            if (const auto* value = std::get_if<std::string>(&entry->value)) {
                constants.used_names.insert(expr.text);
                return *value;
            }
            throw_config_error(path, expr.loc, context + " constant '" + expr.text + "' must be a string");
        }
        return expr.text;
    }
    throw_config_error(path, expr.loc, context + " must be a symbol or string");
}

DistKind parse_dist_kind_name(const std::string& path,
                              const SourceLocation& loc,
                              const std::string& name) {
    if (name == "constant") return DistKind::constant;
    if (name == "uniform") return DistKind::uniform;
    if (name == "normal") return DistKind::normal;
    if (name == "exponential") return DistKind::exponential;
    if (name == "gamma") return DistKind::gamma;
    if (name == "beta") return DistKind::beta;
    throw_config_error(path, loc, "unknown distribution builder '" + name + "'");
}

StatisticKind parse_stat_kind_name(const std::string& path,
                                   const SourceLocation& loc,
                                   const std::string& name) {
    if (name == "mean_fitness") return StatisticKind::mean_fitness;
    if (name == "genetic_load") return StatisticKind::genetic_load;
    if (name == "realized_masking_bonus") return StatisticKind::realized_masking_bonus;
    if (name == "exact_B") return StatisticKind::exact_B;
    if (name == "pairwise_similarity") return StatisticKind::mean_pairwise_haplotypic_similarity;
    if (name == "n_seg") return StatisticKind::n_seg;
    if (name == "n_fixed") return StatisticKind::n_fixed;
    if (name == "genome_words") return StatisticKind::genome_words;
    if (name == "mutation_histogram") return StatisticKind::mutation_histogram;
    if (name == "site_frequency_spectrum") return StatisticKind::site_frequency_spectrum;
    if (name == "nucleotide_diversity") return StatisticKind::nucleotide_diversity;
    if (name == "expected_heterozygosity") return StatisticKind::expected_heterozygosity;
    throw_config_error(path, loc, "unknown statistic name '" + name + "'");
}

HaplotypeSimilarityMetric parse_similarity_metric_name(const std::string& path,
                                                       const SourceLocation& loc,
                                                       const std::string& name) {
    if (name == "jaccard") return HaplotypeSimilarityMetric::jaccard;
    if (name == "dice") return HaplotypeSimilarityMetric::dice;
    if (name == "overlap") return HaplotypeSimilarityMetric::overlap;
    throw_config_error(path, loc, "unknown pairwise-similarity metric '" + name + "'");
}

ParentSamplerBuildMode parse_alias_builder_name(const std::string& path,
                                                const SourceLocation& loc,
                                                const std::string& name) {
    if (name == "auto") return ParentSamplerBuildMode::automatic;
    if (name == "sequential") return ParentSamplerBuildMode::sequential;
    if (name == "parallel") return ParentSamplerBuildMode::parallel;
    if (name == "parallel_psa_plus") return ParentSamplerBuildMode::parallel_psa_plus;
    throw_config_error(
        path,
        loc,
        "unknown alias_builder '" + name
            + "'; expected auto, sequential, parallel, or parallel_psa_plus");
}

void validate_clamp_bounds(const std::string& path,
                           const Expr& expr,
                           double min_value,
                           double max_value,
                           const std::string& context) {
    if (max_value < min_value) {
        throw_config_error(
            path,
            expr.loc,
            context + " has clamp bounds where max < min");
    }
}

DistSpec parse_dist_spec(const std::string& path,
                         const Expr& expr,
                         ConstantScope& constants,
                         const std::string& context) {
    if (expr.kind != Expr::Kind::call) {
        throw_config_error(path, expr.loc, context + " must be a distribution builder call");
    }

    const DistKind kind = parse_dist_kind_name(path, expr.loc, expr.text);
    ArgReader args(path, expr);
    DistSpec spec;
    spec.kind = kind;

    switch (kind) {
        case DistKind::constant: {
            const Expr& value_expr = args.required("value", expr.text);
            const Expr* min_expr = args.optional("min");
            const Expr* max_expr = args.optional("max");
            const double value = resolve_double(path, value_expr, constants, "constant(value)");
            spec.p1 = value;
            spec.p2 = 0.0;
            spec.min_value = (min_expr != nullptr)
                                 ? resolve_double(path, *min_expr, constants, "constant min")
                                 : value;
            spec.max_value = (max_expr != nullptr)
                                 ? resolve_double(path, *max_expr, constants, "constant max")
                                 : value;
            args.require_no_extra(expr.text);
            if (value < spec.min_value || value > spec.max_value) {
                throw_config_error(
                    path,
                    expr.loc,
                    "constant(value, min, max) requires value to lie within [min, max]");
            }
            break;
        }
        case DistKind::uniform: {
            spec.p1 = resolve_double(path,
                                     args.required("min", expr.text),
                                     constants,
                                     "uniform min");
            spec.p2 = resolve_double(path,
                                     args.required("max", expr.text),
                                     constants,
                                     "uniform max");
            const Expr* clamp_min = args.optional("clamp_min");
            const Expr* clamp_max = args.optional("clamp_max");
            spec.min_value = (clamp_min != nullptr)
                                 ? resolve_double(path, *clamp_min, constants, "uniform clamp_min")
                                 : spec.p1;
            spec.max_value = (clamp_max != nullptr)
                                 ? resolve_double(path, *clamp_max, constants, "uniform clamp_max")
                                 : spec.p2;
            args.require_no_extra(expr.text);
            if (spec.p2 < spec.p1) {
                throw_config_error(path, expr.loc, "uniform requires max >= min");
            }
            break;
        }
        case DistKind::normal: {
            spec.p1 = resolve_double(path,
                                     args.required("mean", expr.text),
                                     constants,
                                     "normal mean");
            spec.p2 = resolve_double(path,
                                     args.required("sd", expr.text),
                                     constants,
                                     "normal sd");
            spec.min_value = resolve_double(path,
                                            args.required("min", expr.text),
                                            constants,
                                            "normal min");
            spec.max_value = resolve_double(path,
                                            args.required("max", expr.text),
                                            constants,
                                            "normal max");
            args.require_no_extra(expr.text);
            if (spec.p2 < 0.0) {
                throw_config_error(path, expr.loc, "normal sd must be non-negative");
            }
            break;
        }
        case DistKind::exponential: {
            spec.p1 = resolve_double(path,
                                     args.required("mean", expr.text),
                                     constants,
                                     "exponential mean");
            spec.p2 = 0.0;
            spec.min_value = resolve_double(path,
                                            args.required("min", expr.text),
                                            constants,
                                            "exponential min");
            spec.max_value = resolve_double(path,
                                            args.required("max", expr.text),
                                            constants,
                                            "exponential max");
            args.require_no_extra(expr.text);
            if (spec.p1 < 0.0) {
                throw_config_error(path, expr.loc, "exponential mean must be non-negative");
            }
            break;
        }
        case DistKind::gamma: {
            spec.p1 = resolve_double(path,
                                     args.required("mean", expr.text),
                                     constants,
                                     "gamma mean");
            spec.p2 = resolve_double(path,
                                     args.required("shape", expr.text),
                                     constants,
                                     "gamma shape");
            spec.min_value = resolve_double(path,
                                            args.required("min", expr.text),
                                            constants,
                                            "gamma min");
            spec.max_value = resolve_double(path,
                                            args.required("max", expr.text),
                                            constants,
                                            "gamma max");
            args.require_no_extra(expr.text);
            if (spec.p2 <= 0.0) {
                throw_config_error(path, expr.loc, "gamma shape must be > 0");
            }
            break;
        }
        case DistKind::beta: {
            spec.p1 = resolve_double(path,
                                     args.required("alpha", expr.text),
                                     constants,
                                     "beta alpha");
            spec.p2 = resolve_double(path,
                                     args.required("beta", expr.text),
                                     constants,
                                     "beta beta");
            spec.min_value = resolve_double(path,
                                            args.required("min", expr.text),
                                            constants,
                                            "beta min");
            spec.max_value = resolve_double(path,
                                            args.required("max", expr.text),
                                            constants,
                                            "beta max");
            args.require_no_extra(expr.text);
            if (spec.p1 <= 0.0) {
                throw_config_error(path, expr.loc, "beta alpha must be > 0");
            }
            if (spec.p2 <= 0.0) {
                throw_config_error(path, expr.loc, "beta beta must be > 0");
            }
            break;
        }
    }

    validate_clamp_bounds(path, expr, spec.min_value, spec.max_value, expr.text);
    return spec;
}

DominanceSpec parse_dominance_spec(const std::string& path,
                                   const Expr& expr,
                                   ConstantScope& constants) {
    if (expr.kind != Expr::Kind::call) {
        throw_config_error(path, expr.loc, "dominance must be a builder call");
    }

    ArgReader args(path, expr);
    if (expr.text == "additive") {
        args.require_no_extra(expr.text);
        return DominanceSpec::additive();
    }
    if (expr.text == "fixed") {
        const double h = resolve_double(
            path, args.required("h", expr.text), constants, "fixed dominance h");
        args.require_no_extra(expr.text);
        return DominanceSpec::fixed(h);
    }
    if (expr.text == "distributed") {
        const Expr& distribution_expr = args.required("distribution", expr.text);
        args.require_no_extra(expr.text);
        return DominanceSpec::distributed(
            parse_dist_spec(path, distribution_expr, constants, "distributed dominance"));
    }
    if (expr.text == "linear_from_s") {
        const double intercept = resolve_double(path,
                                                args.required("intercept", expr.text),
                                                constants,
                                                "linear_from_s intercept");
        const double slope = resolve_double(path,
                                            args.required("slope", expr.text),
                                            constants,
                                            "linear_from_s slope");
        const double min_h = resolve_double(path,
                                            args.required("min_h", expr.text),
                                            constants,
                                            "linear_from_s min_h");
        const double max_h = resolve_double(path,
                                            args.required("max_h", expr.text),
                                            constants,
                                            "linear_from_s max_h");
        args.require_no_extra(expr.text);
        if (max_h < min_h) {
            throw_config_error(path, expr.loc, "linear_from_s has max_h < min_h");
        }
        return DominanceSpec::linear_from_s(intercept, slope, min_h, max_h);
    }

    throw_config_error(path, expr.loc, "unknown dominance builder '" + expr.text + "'");
}

std::vector<uint64_t> parse_generation_vector(const std::string& path,
                                              const Expr& expr,
                                              ConstantScope& constants,
                                              const std::string& context) {
    if (expr.kind == Expr::Kind::call && expr.text == "c") {
        ArgReader args(path, expr);
        std::vector<uint64_t> generations;
        generations.reserve(args.positional_count());
        for (const Expr* generation_expr : args.take_remaining_positional()) {
            generations.push_back(resolve_uint64(path, *generation_expr, constants, context));
        }
        args.require_no_extra("c");
        return generations;
    }

    return {resolve_uint64(path, expr, constants, context)};
}

StatisticKind parse_simple_stat_kind(const std::string& path,
                                     const Expr& expr,
                                     const std::string& context) {
    if (expr.kind == Expr::Kind::symbol) {
        return parse_stat_kind_name(path, expr.loc, expr.text);
    }
    if (expr.kind == Expr::Kind::call) {
        if (!expr.args.empty()) {
            throw_config_error(path, expr.loc, context + " does not accept arguments");
        }
        return parse_stat_kind_name(path, expr.loc, expr.text);
    }
    throw_config_error(path, expr.loc, context + " must be a statistic symbol or call");
}

void append_stat_request_specs(const std::string& path,
                               const Expr& expr,
                               ConstantScope& constants,
                               bool every_generation,
                               const std::vector<uint64_t>& generations,
                               std::vector<PendingStatisticSpec>& out_specs) {
    if (expr.kind == Expr::Kind::symbol
        || (expr.kind == Expr::Kind::call && expr.text != "pairwise_similarity")) {
        PendingStatisticSpec spec;
        spec.kind = parse_simple_stat_kind(path, expr, "statistic");
        spec.every_generation = every_generation;
        spec.generations = generations;
        spec.loc = expr.loc;
        out_specs.push_back(std::move(spec));
        return;
    }

    if (expr.kind != Expr::Kind::call) {
        throw_config_error(path, expr.loc, "invalid statistic specification");
    }
    if (expr.text != "pairwise_similarity") {
        throw_config_error(path, expr.loc, "unknown statistic builder '" + expr.text + "'");
    }

    ArgReader args(path, expr);
    const std::string metric_name = resolve_name(
        path, args.required("metric", expr.text), constants, "pairwise_similarity metric");
    args.require_no_extra(expr.text);

    PendingStatisticSpec spec;
    spec.kind = StatisticKind::mean_pairwise_haplotypic_similarity;
    spec.metric = parse_similarity_metric_name(path, expr.loc, metric_name);
    spec.every_generation = every_generation;
    spec.generations = generations;
    spec.loc = expr.loc;
    out_specs.push_back(std::move(spec));
}

uint64_t parse_bounded_generation(const std::string& path,
                                  const Expr& expr,
                                  ConstantScope& constants,
                                  int generation_count,
                                  const std::string& context) {
    const uint64_t generation = resolve_uint64(path, expr, constants, context);
    if (generation == 0u || generation > (uint64_t)generation_count) {
        throw_config_error(
            path,
            expr.loc,
            context + " "
                + std::to_string((unsigned long long)generation)
                + " is outside the valid range [1, G]");
    }
    return generation;
}

std::vector<uint64_t> expand_dense_range(uint64_t start,
                                         uint64_t end) {
    std::vector<uint64_t> generations;
    generations.reserve((size_t)(end - start + 1u));
    for (uint64_t generation = start; generation <= end; ++generation) {
        generations.push_back(generation);
    }
    return generations;
}

std::vector<uint64_t> expand_every_range(uint64_t step,
                                         uint64_t start,
                                         uint64_t end) {
    std::vector<uint64_t> generations;
    if (start > end) return generations;
    const size_t reserve_hint = (size_t)((end - start) / step + 1u);
    generations.reserve(reserve_hint);
    for (uint64_t generation = start; generation <= end; generation += step) {
        generations.push_back(generation);
        if (generation > end - step) break;
    }
    return generations;
}

void parse_stats_section(const std::string& path,
                         const Expr& stats_expr,
                         ConstantScope& constants,
                         int generation_count,
                         LoadedConfig& loaded) {
    const Expr& stats_list = expect_list_call(path, stats_expr, "stats");
    std::vector<PendingStatisticSpec> pending_specs;

    for (const CallArg& entry_arg : stats_list.args) {
        if (entry_arg.has_name) {
            throw_config_error(path,
                               entry_arg.value->loc,
                               "stats entries must be unnamed schedule calls");
        }
        const Expr& entry = *entry_arg.value;
        if (entry.kind != Expr::Kind::call) {
            throw_config_error(
                path,
                entry.loc,
                "stats entries must be always(...), every(...), at(...), up_to(...), at_after(...), every_up_to(...), every_at_after(...), range(...), or every_range(...)");
        }

        if (entry.text == "always") {
            ArgReader args(path, entry);
            if (args.positional_count() == 0u) {
                throw_config_error(path, entry.loc, "always(...) requires at least one statistic");
            }
            for (const Expr* stat_expr : args.take_remaining_positional()) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    true,
                    {},
                    pending_specs);
            }
            args.require_no_extra("always");
            continue;
        }

        if (entry.text == "every") {
            ArgReader args(path, entry);
            const uint64_t step = resolve_uint64(path,
                                                 args.required("step", "every"),
                                                 constants,
                                                 "every step");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "every(step, ...) requires a step and at least one statistic");
            }
            if (step == 0u)
                throw_config_error(path, entry.loc, "every step must be positive");
            if (step > (uint64_t)generation_count) {
                loaded.warnings.push_back(format_warning(
                    path,
                    entry.loc,
                    "every(" + std::to_string((unsigned long long)step)
                        + ", ...) never fires because G="
                        + std::to_string(generation_count)));
            }
            const std::vector<uint64_t> generations =
                expand_every_range(step, step, (uint64_t)generation_count);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("every");
            continue;
        }

        if (entry.text == "at") {
            ArgReader args(path, entry);
            const Expr& generations_expr = args.required("generations", "at");
            std::vector<uint64_t> generations = parse_generation_vector(
                path, generations_expr, constants, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "at(generations, ...) requires generations and at least one statistic");
            }
            if (generations.empty()) {
                throw_config_error(path, entry.loc, "at(...) requires at least one generation");
            }
            for (size_t generation_index = 0u; generation_index < generations.size(); ++generation_index) {
                const uint64_t generation = generations[generation_index];
                if (generation == 0u || generation > (uint64_t)generation_count) {
                    throw_config_error(
                        path,
                        generations_expr.loc,
                        "generation " + std::to_string((unsigned long long)generation)
                            + " is outside the valid range [1, G]");
                }
            }
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("at");
            continue;
        }

        if (entry.text == "up_to") {
            ArgReader args(path, entry);
            const uint64_t end = parse_bounded_generation(
                path, args.required("end", "up_to"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "up_to(end, ...) requires an end generation and at least one statistic");
            }
            const std::vector<uint64_t> generations = expand_dense_range(1u, end);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("up_to");
            continue;
        }

        if (entry.text == "at_after") {
            ArgReader args(path, entry);
            const uint64_t start = parse_bounded_generation(
                path, args.required("start", "at_after"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "at_after(start, ...) requires a start generation and at least one statistic");
            }
            const std::vector<uint64_t> generations =
                expand_dense_range(start, (uint64_t)generation_count);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("at_after");
            continue;
        }

        if (entry.text == "every_up_to") {
            ArgReader args(path, entry);
            const uint64_t step = resolve_uint64(
                path, args.required("step", "every_up_to"), constants, "every_up_to step");
            const uint64_t end = parse_bounded_generation(
                path, args.required("end", "every_up_to"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "every_up_to(step, end, ...) requires a step, an end generation, and at least one statistic");
            }
            if (step == 0u) {
                throw_config_error(path, entry.loc, "every_up_to step must be positive");
            }
            const std::vector<uint64_t> generations = expand_every_range(step, step, end);
            if (generations.empty()) {
                loaded.warnings.push_back(format_warning(
                    path,
                    entry.loc,
                    "every_up_to(" + std::to_string((unsigned long long)step)
                        + ", " + std::to_string((unsigned long long)end)
                        + ", ...) never fires because step > end"));
            }
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("every_up_to");
            continue;
        }

        if (entry.text == "every_at_after") {
            ArgReader args(path, entry);
            const uint64_t step = resolve_uint64(
                path, args.required("step", "every_at_after"), constants, "every_at_after step");
            const uint64_t start = parse_bounded_generation(
                path, args.required("start", "every_at_after"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "every_at_after(step, start, ...) requires a step, a start generation, and at least one statistic");
            }
            if (step == 0u) {
                throw_config_error(path, entry.loc, "every_at_after step must be positive");
            }
            const std::vector<uint64_t> generations =
                expand_every_range(step, start, (uint64_t)generation_count);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("every_at_after");
            continue;
        }

        if (entry.text == "range") {
            ArgReader args(path, entry);
            const uint64_t start = parse_bounded_generation(
                path, args.required("start", "range"), constants, generation_count, "generation");
            const uint64_t end = parse_bounded_generation(
                path, args.required("end", "range"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "range(start, end, ...) requires a start generation, an end generation, and at least one statistic");
            }
            if (end < start) {
                throw_config_error(path, entry.loc, "range(start, end, ...) requires end >= start");
            }
            const std::vector<uint64_t> generations = expand_dense_range(start, end);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("range");
            continue;
        }

        if (entry.text == "every_range") {
            ArgReader args(path, entry);
            const uint64_t step = resolve_uint64(
                path, args.required("step", "every_range"), constants, "every_range step");
            const uint64_t start = parse_bounded_generation(
                path, args.required("start", "every_range"), constants, generation_count, "generation");
            const uint64_t end = parse_bounded_generation(
                path, args.required("end", "every_range"), constants, generation_count, "generation");
            const std::vector<const Expr*> stats = args.take_remaining_positional();
            if (stats.empty()) {
                throw_config_error(path, entry.loc, "every_range(step, start, end, ...) requires a step, a start generation, an end generation, and at least one statistic");
            }
            if (step == 0u) {
                throw_config_error(path, entry.loc, "every_range step must be positive");
            }
            if (end < start) {
                throw_config_error(path, entry.loc, "every_range(step, start, end, ...) requires end >= start");
            }
            const std::vector<uint64_t> generations = expand_every_range(step, start, end);
            for (const Expr* stat_expr : stats) {
                append_stat_request_specs(
                    path,
                    *stat_expr,
                    constants,
                    false,
                    generations,
                    pending_specs);
            }
            args.require_no_extra("every_range");
            continue;
        }

        throw_config_error(path, entry.loc, "unknown stats schedule builder '" + entry.text + "'");
    }

    for (const PendingStatisticSpec& pending : pending_specs) {
        StatisticRequest request;
        request.kind = pending.kind;
        request.similarity_metric = pending.metric;
        if (!pending.every_generation) {
            request.generations = pending.generations;
        }
        loaded.params.statistic_requests.push_back(std::move(request));
    }

    loaded.has_statistics = !loaded.params.statistic_requests.empty();
}

std::unordered_map<std::string, const Expr*> named_entries(const std::string& path,
                                                           const Expr& list_expr,
                                                           const std::string& context) {
    std::unordered_map<std::string, const Expr*> entries;
    entries.reserve(list_expr.args.size());
    for (const CallArg& arg : list_expr.args) {
        if (!arg.has_name) {
            throw_config_error(path,
                               arg.value->loc,
                               context + " entries must be named");
        }
        if (entries.count(arg.name) != 0u) {
            throw_config_error(path,
                               arg.value->loc,
                               "duplicate " + context + " entry '" + arg.name + "'");
        }
        entries[arg.name] = arg.value.get();
    }
    return entries;
}

void require_allowed_named_fields(const std::string& path,
                                  const std::unordered_map<std::string, const Expr*>& entries,
                                  const std::vector<std::string>& required,
                                  const std::vector<std::string>& optional,
                                  const std::string& context,
                                  const SourceLocation& loc) {
    for (const std::string& field : required) {
        if (entries.count(field) == 0u) {
            throw_config_error(path, loc, context + " is missing required field '" + field + "'");
        }
    }
    for (const auto& entry : entries) {
        const bool in_required =
            std::find(required.begin(), required.end(), entry.first) != required.end();
        const bool in_optional =
            std::find(optional.begin(), optional.end(), entry.first) != optional.end();
        if (!in_required && !in_optional) {
            throw_config_error(path,
                               entry.second->loc,
                               context + " has unknown field '" + entry.first + "'");
        }
    }
}

void parse_mutation_types(const std::string& path,
                          const Expr& expr,
                          ConstantScope& constants,
                          LoadedConfig& loaded,
                          std::unordered_map<std::string, uint32_t>& mutation_indices) {
    const Expr& mutation_list = expect_list_call(path, expr, "config$mutation_types");
    const auto mutation_entries = named_entries(path, mutation_list, "mutation_types");
    if (mutation_entries.empty()) {
        throw_config_error(path, mutation_list.loc, "config$mutation_types must not be empty");
    }

    for (const CallArg& arg : mutation_list.args) {
        const std::string& name = arg.name;
        const Expr& mutation_expr = *arg.value;
        const Expr& mutation_spec = expect_list_call(
            path,
            mutation_expr,
            "mutation_types$" + name);
        const auto fields = named_entries(path, mutation_spec, "mutation type");
        require_allowed_named_fields(path,
                                     fields,
                                     {"selection", "dominance"},
                                     {},
                                     "mutation_types$" + name,
                                     mutation_spec.loc);

        MutationTypeSpec spec;
        spec.selection = parse_dist_spec(
            path, *fields.at("selection"), constants, "mutation_types$" + name + "$selection");
        spec.dominance = parse_dominance_spec(path, *fields.at("dominance"), constants);

        mutation_indices[name] = (uint32_t)loaded.params.mutation_types.size();
        loaded.params.mutation_types.push_back(std::move(spec));
    }
}

void parse_region_types(const std::string& path,
                        const Expr& expr,
                        ConstantScope& constants,
                        const std::unordered_map<std::string, uint32_t>& mutation_indices,
                        LoadedConfig& loaded,
                        std::unordered_map<std::string, uint32_t>& region_indices) {
    const Expr& region_list = expect_list_call(path, expr, "config$region_types");
    const auto region_entries = named_entries(path, region_list, "region_types");
    if (region_entries.empty()) {
        throw_config_error(path, region_list.loc, "config$region_types must not be empty");
    }

    for (const CallArg& arg : region_list.args) {
        const std::string& name = arg.name;
        const Expr& region_expr = *arg.value;
        const Expr& region_spec = expect_list_call(path, region_expr, "region_types$" + name);
        const auto fields = named_entries(path, region_spec, "region type");
        require_allowed_named_fields(path,
                                     fields,
                                     {"mutation_scale", "weights"},
                                     {},
                                     "region_types$" + name,
                                     region_spec.loc);

        RegionTypeSpec spec;
        spec.mutation_scale = resolve_double(
            path, *fields.at("mutation_scale"), constants, "region_types$" + name + "$mutation_scale");
        if (spec.mutation_scale < 0.0) {
            throw_config_error(path,
                               fields.at("mutation_scale")->loc,
                               "region_types$" + name + "$mutation_scale must be non-negative");
        }

        const Expr& weights_expr = expect_call_named(
            path, *fields.at("weights"), "c", "region_types$" + name + "$weights");
        if (weights_expr.args.empty()) {
            throw_config_error(path, weights_expr.loc, "weights vectors must not be empty");
        }
        double total_weight = 0.0;
        for (const CallArg& weight_arg : weights_expr.args) {
            if (!weight_arg.has_name) {
                throw_config_error(path,
                                   weight_arg.value->loc,
                                   "weights must use named entries like c(deleterious = 8)");
            }
            const auto mutation_it = mutation_indices.find(weight_arg.name);
            if (mutation_it == mutation_indices.end()) {
                throw_config_error(path,
                                   weight_arg.value->loc,
                                   "weights reference unknown mutation type '" + weight_arg.name + "'");
            }
            const double weight = resolve_double(
                path, *weight_arg.value, constants, "weight for mutation type '" + weight_arg.name + "'");
            if (weight <= 0.0) {
                throw_config_error(path,
                                   weight_arg.value->loc,
                                   "weights for mutation type '" + weight_arg.name + "' must be > 0");
            }
            total_weight += weight;
            spec.mutation_types.push_back({mutation_it->second, weight});
        }
        if (total_weight <= 0.0) {
            throw_config_error(path, weights_expr.loc, "weights must sum to > 0");
        }

        region_indices[name] = (uint32_t)loaded.params.mutation_region_types.size();
        loaded.params.mutation_region_types.push_back(std::move(spec));
    }
}

RecIntervalSpec parse_interval_spec(const std::string& path,
                                    const Expr& expr,
                                    ConstantScope& constants) {
    const Expr& interval_expr = expect_call_named(path, expr, "interval", "recombination interval");
    ArgReader args(path, interval_expr);
    RecIntervalSpec interval;
    interval.start = (uint32_t)resolve_uint64(
        path,
        args.required("start", "interval"),
        constants,
        "interval start");
    interval.end = (uint32_t)resolve_uint64(
        path,
        args.required("end", "interval"),
        constants,
        "interval end");
    interval.rate_scale = resolve_double(
        path,
        args.required("rate_scale", "interval"),
        constants,
        "interval rate_scale");
    args.require_no_extra("interval");
    if (interval.end <= interval.start) {
        throw_config_error(path, interval_expr.loc, "interval end must be greater than start");
    }
    if (interval.rate_scale < 0.0) {
        throw_config_error(path, interval_expr.loc, "interval rate_scale must be non-negative");
    }
    return interval;
}

ChromosomeRegionSpec parse_region_placement(const std::string& path,
                                            const Expr& expr,
                                            ConstantScope& constants,
                                            const std::unordered_map<std::string, uint32_t>& region_indices) {
    const Expr& region_expr = expect_call_named(path, expr, "region", "chromosome region");
    ArgReader args(path, region_expr);
    const std::string region_name = resolve_name(
        path,
        args.required("region_type", "region"),
        constants,
        "region type");
    const auto region_it = region_indices.find(region_name);
    if (region_it == region_indices.end()) {
        throw_config_error(path, region_expr.loc, "unknown region type '" + region_name + "'");
    }
    ChromosomeRegionSpec region;
    region.region_type_index = region_it->second;
    region.start = (uint32_t)resolve_uint64(
        path,
        args.required("start", "region"),
        constants,
        "region start");
    region.end = (uint32_t)resolve_uint64(
        path,
        args.required("end", "region"),
        constants,
        "region end");
    args.require_no_extra("region");
    if (region.end <= region.start) {
        throw_config_error(path, region_expr.loc, "region end must be greater than start");
    }
    return region;
}

void validate_coverage(const std::string& path,
                       const SourceLocation& loc,
                       const std::string& context,
                       uint32_t length,
                       const std::vector<RecIntervalSpec>& intervals) {
    if (intervals.empty())
        throw_config_error(path, loc, context + " must contain at least one interval");

    uint32_t cursor = 0u;
    for (const RecIntervalSpec& interval : intervals) {
        if (interval.start != cursor) {
            throw_config_error(path,
                               loc,
                               context + " must be contiguous from 0 without gaps or overlaps");
        }
        if (interval.end > length) {
            throw_config_error(path, loc, context + " extends past chromosome length");
        }
        cursor = interval.end;
    }
    if (cursor != length) {
        throw_config_error(path, loc, context + " must cover the full chromosome length");
    }
}

void validate_region_coverage(const std::string& path,
                              const SourceLocation& loc,
                              const std::string& context,
                              uint32_t length,
                              const std::vector<ChromosomeRegionSpec>& regions) {
    if (regions.empty())
        throw_config_error(path, loc, context + " must contain at least one region");

    uint32_t cursor = 0u;
    for (const ChromosomeRegionSpec& region : regions) {
        if (region.start != cursor) {
            throw_config_error(path,
                               loc,
                               context + " must be contiguous from 0 without gaps or overlaps");
        }
        if (region.end > length) {
            throw_config_error(path, loc, context + " extends past chromosome length");
        }
        cursor = region.end;
    }
    if (cursor != length) {
        throw_config_error(path, loc, context + " must cover the full chromosome length");
    }
}

void parse_chromosomes(const std::string& path,
                       const Expr& expr,
                       ConstantScope& constants,
                       const std::unordered_map<std::string, uint32_t>& region_indices,
                       LoadedConfig& loaded) {
    const Expr& chromosome_list = expect_list_call(path, expr, "config$chromosomes");
    const auto chromosome_entries = named_entries(path, chromosome_list, "chromosomes");
    if (chromosome_entries.empty()) {
        throw_config_error(path, chromosome_list.loc, "config$chromosomes must not be empty");
    }

    for (const CallArg& arg : chromosome_list.args) {
        const std::string& name = arg.name;
        const Expr& chromosome_expr = *arg.value;
        const Expr& chromosome_spec = expect_list_call(path, chromosome_expr, "chromosomes$" + name);
        const auto fields = named_entries(path, chromosome_spec, "chromosome");
        require_allowed_named_fields(path,
                                     fields,
                                     {"length", "recombination_intervals", "regions"},
                                     {},
                                     "chromosomes$" + name,
                                     chromosome_spec.loc);

        ChromosomeSpec chromosome;
        chromosome.length = (uint32_t)resolve_uint64(
            path, *fields.at("length"), constants, "chromosomes$" + name + "$length");
        if (chromosome.length == 0u) {
            throw_config_error(path,
                               fields.at("length")->loc,
                               "chromosomes$" + name + "$length must be positive");
        }

        const Expr& intervals_expr = expect_list_call(
            path,
            *fields.at("recombination_intervals"),
            "chromosomes$" + name + "$recombination_intervals");
        for (const CallArg& interval_arg : intervals_expr.args) {
            if (interval_arg.has_name) {
                throw_config_error(path,
                                   interval_arg.value->loc,
                                   "recombination_intervals entries must be unnamed");
            }
            chromosome.recombination_map.push_back(
                parse_interval_spec(path, *interval_arg.value, constants));
        }
        validate_coverage(path,
                          intervals_expr.loc,
                          "chromosomes$" + name + "$recombination_intervals",
                          chromosome.length,
                          chromosome.recombination_map);

        const Expr& regions_expr = expect_list_call(
            path,
            *fields.at("regions"),
            "chromosomes$" + name + "$regions");
        for (const CallArg& region_arg : regions_expr.args) {
            if (region_arg.has_name) {
                throw_config_error(path,
                                   region_arg.value->loc,
                                   "regions entries must be unnamed");
            }
            chromosome.regions.push_back(
                parse_region_placement(path, *region_arg.value, constants, region_indices));
        }
        validate_region_coverage(path,
                                 regions_expr.loc,
                                 "chromosomes$" + name + "$regions",
                                 chromosome.length,
                                 chromosome.regions);

        loaded.params.chromosomes.push_back(std::move(chromosome));
    }
}

void parse_runtime_config(const std::string& path,
                          const Expr& expr,
                          ConstantScope& constants,
                          SimParams& params) {
    const Expr& runtime_list = expect_list_call(path, expr, "config$runtime");
    const auto fields = named_entries(path, runtime_list, "runtime");
    require_allowed_named_fields(path,
                                 fields,
                                 {},
                                 {"alias_builder", "profile"},
                                 "config$runtime",
                                 runtime_list.loc);

    const auto alias_it = fields.find("alias_builder");
    if (alias_it != fields.end()) {
        const std::string name = resolve_name(
            path, *alias_it->second, constants, "config$runtime$alias_builder");
        params.parent_sampler_build_mode =
            parse_alias_builder_name(path, alias_it->second->loc, name);
    }

    const auto profile_it = fields.find("profile");
    if (profile_it != fields.end()) {
        params.enable_profiling = resolve_bool(
            path, *profile_it->second, constants, "config$runtime$profile");
    }
}

void parse_config_section(const std::string& path,
                          const Expr& config_expr,
                          ConstantScope& constants,
                          LoadedConfig& loaded) {
    const Expr& config_list = expect_list_call(path, config_expr, "config");
    const auto fields = named_entries(path, config_list, "config");
    require_allowed_named_fields(path,
                                 fields,
                                 {"N",
                                  "G",
                                  "mu",
                                  "rho",
                                  "seed",
                                  "threads",
                                  "mutation_types",
                                  "region_types",
                                  "chromosomes"},
                                 {"runtime"},
                                 "config",
                                 config_list.loc);

    loaded.params.N = resolve_int(path, *fields.at("N"), constants, "config$N");
    loaded.params.G = resolve_int(path, *fields.at("G"), constants, "config$G");
    loaded.params.mu = resolve_double(path, *fields.at("mu"), constants, "config$mu");
    loaded.params.rho = resolve_double(path, *fields.at("rho"), constants, "config$rho");
    loaded.params.seed = resolve_uint64(path, *fields.at("seed"), constants, "config$seed");
    loaded.params.threads = resolve_int(path, *fields.at("threads"), constants, "config$threads");

    if (loaded.params.N <= 0) {
        throw_config_error(path, fields.at("N")->loc, "config$N must be positive");
    }
    if (loaded.params.G <= 0) {
        throw_config_error(path, fields.at("G")->loc, "config$G must be positive");
    }
    if (loaded.params.mu < 0.0) {
        throw_config_error(path, fields.at("mu")->loc, "config$mu must be non-negative");
    }
    if (loaded.params.rho < 0.0) {
        throw_config_error(path, fields.at("rho")->loc, "config$rho must be non-negative");
    }
    if (loaded.params.threads < 0) {
        throw_config_error(path, fields.at("threads")->loc, "config$threads must be non-negative");
    }

    const auto runtime_it = fields.find("runtime");
    if (runtime_it != fields.end()) {
        parse_runtime_config(path, *runtime_it->second, constants, loaded.params);
    }

    std::unordered_map<std::string, uint32_t> mutation_indices;
    parse_mutation_types(path,
                         *fields.at("mutation_types"),
                         constants,
                         loaded,
                         mutation_indices);

    std::unordered_map<std::string, uint32_t> region_indices;
    parse_region_types(path,
                       *fields.at("region_types"),
                       constants,
                       mutation_indices,
                       loaded,
                       region_indices);

    parse_chromosomes(path,
                      *fields.at("chromosomes"),
                      constants,
                      region_indices,
                      loaded);
}

ConstantScope parse_constants_section(const std::string& path, const Expr& constants_expr) {
    const Expr& constants_list = expect_list_call(path, constants_expr, "constants");
    ConstantScope constants;
    constants.values.reserve(constants_list.args.size());
    constants.used_names.reserve(constants_list.args.size());
    for (const CallArg& arg : constants_list.args) {
        if (!arg.has_name) {
            throw_config_error(path,
                               arg.value->loc,
                               "constants entries must be named like name = value");
        }
        if (constants.values.count(arg.name) != 0u) {
            throw_config_error(path,
                               arg.value->loc,
                               "duplicate constant '" + arg.name + "'");
        }
        constants.values.emplace(
            arg.name,
            ConstantEntry{
                evaluate_constant_literal(path, *arg.value),
                arg.value->loc
            });
    }
    return constants;
}

}  // namespace

LoadedConfig load_config_file(const std::string& path) {
    Tokenizer tokenizer(path, read_file_contents(path));
    Parser parser(path, tokenizer.tokenize());
    const std::vector<Assignment> assignments = parser.parse_program();

    const Assignment* constants_assignment = nullptr;
    const Assignment* config_assignment = nullptr;
    const Assignment* stats_assignment = nullptr;

    for (const Assignment& assignment : assignments) {
        if (assignment.name == "constants") {
            if (constants_assignment != nullptr) {
                throw_config_error(path, assignment.loc, "duplicate top-level section 'constants'");
            }
            constants_assignment = &assignment;
            continue;
        }
        if (assignment.name == "config") {
            if (config_assignment != nullptr) {
                throw_config_error(path, assignment.loc, "duplicate top-level section 'config'");
            }
            config_assignment = &assignment;
            continue;
        }
        if (assignment.name == "stats") {
            if (stats_assignment != nullptr) {
                throw_config_error(path, assignment.loc, "duplicate top-level section 'stats'");
            }
            stats_assignment = &assignment;
            continue;
        }

        throw_config_error(
            path,
            assignment.loc,
            "unknown top-level section '" + assignment.name
                + "'; expected only constants, config, and stats");
    }

    if (constants_assignment == nullptr) {
        throw std::runtime_error(path + ": missing required top-level section 'constants'");
    }
    if (config_assignment == nullptr) {
        throw std::runtime_error(path + ": missing required top-level section 'config'");
    }
    if (stats_assignment == nullptr) {
        throw std::runtime_error(path + ": missing required top-level section 'stats'");
    }

    LoadedConfig loaded;
    ConstantScope constants = parse_constants_section(path, *constants_assignment->value);
    parse_config_section(path, *config_assignment->value, constants, loaded);
    parse_stats_section(path,
                        *stats_assignment->value,
                        constants,
                        loaded.params.G,
                        loaded);

    for (const auto& entry : constants.values) {
        if (constants.used_names.count(entry.first) == 0u) {
            loaded.warnings.push_back(format_warning(
                path,
                entry.second.loc,
                "unused constant '" + entry.first + "'"));
        }
    }

    return loaded;
}
