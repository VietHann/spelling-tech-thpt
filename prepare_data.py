# prepare_data.py
# -*- coding: utf-8 -*-
"""
Script to prepare data artifacts for advanced corrector:
1. Vietnamese lexicon (from Hunspell, Wikipedia, etc.)
2. Unigram/bigram frequencies
3. KenLM 5-gram model (Phase 2)
4. Toneless-to-toned mapping (Phase 2)
"""

import os
import re
import json
from collections import Counter, defaultdict
from typing import Set, Dict, List
import unicodedata

# =========================
# 1. BUILD LEXICON
# =========================

def normalize_word(word: str) -> str:
    """Normalize word: lowercase, NFC, strip"""
    word = unicodedata.normalize('NFC', word)
    word = word.lower().strip()
    return word

def build_lexicon_from_hunspell(hunspell_dic_path: str) -> Set[str]:
    """
    Build lexicon from Hunspell .dic file
    
    Hunspell format:
    - First line: word count
    - Following lines: word/flags
    
    Download Vietnamese Hunspell from:
    https://github.com/LibreOffice/dictionaries/tree/master/vi
    """
    lexicon = set()
    
    if not os.path.exists(hunspell_dic_path):
        print(f"[Warning] Hunspell file not found: {hunspell_dic_path}")
        return lexicon
    
    with open(hunspell_dic_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Skip first line (word count)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by '/' to remove flags
            word = line.split('/')[0].strip()
            
            if word:
                word = normalize_word(word)
                if word:
                    lexicon.add(word)
    
    print(f"[Hunspell] Loaded {len(lexicon)} words")
    return lexicon

def build_lexicon_from_text_corpus(corpus_path: str, min_freq: int = 2) -> Set[str]:
    """
    Build lexicon from plain text corpus
    Extract words that appear at least min_freq times
    """
    word_counts = Counter()
    
    if not os.path.exists(corpus_path):
        print(f"[Warning] Corpus file not found: {corpus_path}")
        return set()
    
    # Regex to extract Vietnamese words (syllables)
    word_pattern = re.compile(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+', re.IGNORECASE)
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = unicodedata.normalize('NFC', line)
            words = word_pattern.findall(line.lower())
            word_counts.update(words)
    
    # Filter by frequency
    lexicon = {word for word, count in word_counts.items() if count >= min_freq}
    
    print(f"[Corpus] Loaded {len(lexicon)} words (min_freq={min_freq})")
    return lexicon

def build_lexicon_from_wikipedia_dump(wiki_dump_path: str, max_lines: int = 100000) -> Set[str]:
    """
    Build lexicon from Wikipedia dump (plain text)
    
    Download Vietnamese Wikipedia dump:
    https://dumps.wikimedia.org/viwiki/latest/
    Extract with wikiextractor: https://github.com/attardi/wikiextractor
    """
    return build_lexicon_from_text_corpus(wiki_dump_path, min_freq=3)

def merge_lexicons(*lexicons: Set[str]) -> Set[str]:
    """Merge multiple lexicons"""
    merged = set()
    for lex in lexicons:
        merged.update(lex)
    return merged

def save_lexicon(lexicon: Set[str], output_path: str):
    """Save lexicon to file (one word per line)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(lexicon):
            f.write(word + '\n')
    
    print(f"[Lexicon] Saved {len(lexicon)} words to {output_path}")

# =========================
# 2. BUILD FREQUENCY DICT
# =========================

def build_frequency_dict(corpus_path: str, output_path: str, top_n: int = 100000):
    """
    Build unigram frequency dictionary from corpus
    
    Output format: JSON
    {
        "word1": count1,
        "word2": count2,
        ...
    }
    """
    word_counts = Counter()
    
    if not os.path.exists(corpus_path):
        print(f"[Warning] Corpus file not found: {corpus_path}")
        return
    
    word_pattern = re.compile(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+', re.IGNORECASE)
    
    print(f"[FreqDict] Processing corpus: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"  Processed {i} lines...")
            
            line = unicodedata.normalize('NFC', line)
            words = word_pattern.findall(line.lower())
            word_counts.update(words)
    
    # Keep top N
    top_words = dict(word_counts.most_common(top_n))
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(top_words, f, ensure_ascii=False, indent=2)
    
    print(f"[FreqDict] Saved {len(top_words)} words to {output_path}")

# =========================
# 3. SAMPLE LEXICON (for testing)
# =========================

def create_sample_lexicon(output_path: str):
    """
    Create a small sample lexicon for testing
    Contains common Vietnamese words
    """
    sample_words = [
        # Common words
        "tôi", "bạn", "anh", "chị", "em", "ông", "bà",
        "là", "có", "không", "được", "rất", "nhiều", "ít",
        "đi", "đến", "về", "ra", "vào", "lên", "xuống",
        "làm", "học", "chơi", "ăn", "uống", "ngủ", "nghỉ",
        "nhà", "trường", "công ty", "phòng", "lớp",
        "người", "bạn bè", "gia đình", "con", "cha", "mẹ",
        "ngày", "tháng", "năm", "tuần", "giờ", "phút",
        "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
        "việt nam", "hà nội", "sài gòn", "đà nẵng",
        "xin chào", "cảm ơn", "tạm biệt", "xin lỗi",
        # Common verbs
        "nói", "viết", "đọc", "nghe", "nhìn", "thấy",
        "biết", "hiểu", "nghĩ", "muốn", "cần", "phải",
        "cho", "lấy", "đưa", "mang", "cầm", "để",
        # Common adjectives
        "tốt", "xấu", "đẹp", "to", "nhỏ", "cao", "thấp",
        "nóng", "lạnh", "ấm", "mát", "nhanh", "chậm",
        "mới", "cũ", "trẻ", "già", "khỏe", "yếu",
        # Common nouns
        "nước", "đất", "trời", "mặt trời", "mặt trăng",
        "cây", "hoa", "cỏ", "rừng", "núi", "sông", "biển",
        "xe", "máy bay", "tàu", "thuyền", "đường",
        "sách", "bút", "giấy", "bàn", "ghế", "cửa", "cửa sổ",
        "áo", "quần", "giày", "mũ", "túi",
        "cơm", "phở", "bánh", "nước", "trà", "cà phê",
        # Technology
        "máy tính", "điện thoại", "internet", "email", "website",
        "facebook", "google", "youtube", "zalo",
        # Common phrases
        "làm việc", "đi học", "về nhà", "ăn cơm", "uống nước",
        "chơi game", "xem phim", "nghe nhạc", "đọc sách",
    ]
    
    # Normalize and deduplicate
    lexicon = set()
    for word in sample_words:
        word = normalize_word(word)
        if word:
            lexicon.add(word)
    
    save_lexicon(lexicon, output_path)

# =========================
# MAIN
# =========================

def main():
    """
    Main function to prepare data
    
    Usage:
        python prepare_data.py
    
    This will create:
        - data/vi_lexicon.txt (sample lexicon for testing)
        - data/vi_freq.json (if corpus provided)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for advanced corrector")
    parser.add_argument("--hunspell_dic", type=str, help="Path to Hunspell .dic file")
    parser.add_argument("--corpus", type=str, help="Path to plain text corpus")
    parser.add_argument("--wiki_dump", type=str, help="Path to Wikipedia dump (plain text)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--create_sample", action="store_true", help="Create sample lexicon for testing")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    lexicons = []
    
    # Build from Hunspell
    if args.hunspell_dic:
        lex = build_lexicon_from_hunspell(args.hunspell_dic)
        lexicons.append(lex)
    
    # Build from corpus
    if args.corpus:
        lex = build_lexicon_from_text_corpus(args.corpus, min_freq=2)
        lexicons.append(lex)
        
        # Also build frequency dict
        freq_output = os.path.join(args.output_dir, "vi_freq.json")
        build_frequency_dict(args.corpus, freq_output, top_n=100000)
    
    # Build from Wikipedia
    if args.wiki_dump:
        lex = build_lexicon_from_wikipedia_dump(args.wiki_dump, max_lines=100000)
        lexicons.append(lex)
    
    # Merge and save
    if lexicons:
        merged = merge_lexicons(*lexicons)
        output_path = os.path.join(args.output_dir, "vi_lexicon.txt")
        save_lexicon(merged, output_path)
    
    # Create sample lexicon
    if args.create_sample or not lexicons:
        print("\n[Sample] Creating sample lexicon for testing...")
        sample_output = os.path.join(args.output_dir, "vi_lexicon.txt")
        create_sample_lexicon(sample_output)
        print(f"\n✓ Sample lexicon created at: {sample_output}")
        print("  This is a small lexicon for testing. For production, use --hunspell_dic or --corpus")

if __name__ == "__main__":
    main()

