# LDA + идентификация цитат
LDA analysis (including topic intensity evolution) and quotation identification (for long texts in English and Russian)
Анализ LDA (включая эволюцию интенсивности тем) и идентификация цитат (для длинных текстов на английском и русском языках)

**I. LDA Topic Modeling for Cross-Country Media Discourse Analysis**  
This Python script performs comprehensive LDA topic modeling on media texts from three countries (China, Russia, USA) using Gensim. Key features:  

1. **Multilingual Analysis**  
   - Parallel processing pipelines for Chinese (CN), Russian (RU), and English (US) datasets  
   - Language-specific parameter configurations  

2. **Automated Workflow**  
   - Data loading & preprocessing (token filtering, date handling)  
   - Word2Vec semantic filtering  
   - Dynamic corpus construction  
   - Multicore LDA implementation  

3. **Model Optimization**  
   - Automated topic number selection (2-15 range)  
   - Coherence-perplexity evaluation  
   - Random seed control for reproducibility  

4. **Visual Outputs**  
   - Interactive pyLDAvis bubble charts  
   - Topic strength time trends (2012-2024)  
   - Word clouds and semantic networks  
   - Document-topic matrices  

5. **Technical Specifications**  
   - Input: CSV files with `tokens` and `date` columns  
   - Output: Excel/HTML visualizations in country-specific directories  
   - Dependencies: Gensim, pyLDAvis, WordCloud, PyVis  

**Usage**  
1. Configure input paths and output directories
2. Text preprocessing: removing stop words and restoring parts of speech...
3. Set language-specific parameters in `PARAMS`/`FILTER_SETTINGS`  
4. Execute `main()` for automated processing  

The pipeline enables comparative analysis of media discourse patterns across geopolitical contexts using computational linguistics methods.  


# II. Reporting Verbs Analysis for Cross-Linguistic Media Discourse
Here's a concise English README for the reporting verbs analysis scripts:

---

**II.Reporting Verbs Analysis for Cross-Linguistic Media Discourse**  
This Python toolkit analyzes reporting verbs and speech representation patterns in multilingual media texts. It features:

### Core Functionality
1. **Multilingual Processing Pipelines**
   - English/Chinese: Uses spaCy for syntactic parsing
   - Russian: Leverages pymorphy2 for lemmatization
   - Shared reporting verb lexicons with polarity labels

2. **Speech Representation Detection**
   - **Direct quotes**: Identifies quoted content with contextual verb attribution
   - **Indirect quotes**: Detects complement clauses and reporting phrases
   - **Russian-specific**: Handles Cyrillic quotes and "что" constructions

3. **Key Features**
   - Customizable verb/phrase dictionaries
   - Polarity classification (positive/negative/neutral)
   - Sentence-level granularity
   - Automated Excel reporting

### Technical Implementation
```bash
# English/Chinese processing
python reporting_verbs_диссертация.py --input china_yuanwen.csv

# Russian processing
python reporting_verbs_диссертация.py --input RU_yuanwen.csv

# Short quote filter (post-processing)
python quote_filter.py --input result_RU.xlsx
```

### Outputs
- `result_XX.xlsx`: Sentence-level verb annotations
- `report_XX.xlsx`: Statistical summaries:
  - Verb frequency distributions
  - Polarity breakdowns
  - Direct/indirect speech ratios
- `filtered_quotes.xlsx`: Quotes with ≤3 words (RU)

### Dependencies
- spaCy (`en_core_web_lg`)
- pymorphy2 (Russian)
- pandas, regex, tqdm




