import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

# === CONFIGURATION ===
model_path = "./fine_tuned_model_with_context"  # Path to your fine-tuned model
raw_text = """# Rest of Asia Pacific

Rest of Asia Pacific net sales increased during 2024 compared to 2023 due primarily to higher net sales of Services. The weakness in foreign currencies relative to the U.S. dollar had a net unfavorable year-over-year impact on Rest of Asia Pacific net sales during 2024.

Apple Inc. | 2024 Form 10-K | 22
# Products and Services Performance

| |2024|Change|2023|Change|2022|
|---|---|---|---|---|---|
|iPhone|$ 201,183|— %|$ 200,583|(2)%|$ 205,489|
|Mac|$ 29,984|2 %|$ 29,357|(27)%|$ 40,177|
|iPad|$ 26,694|(6)%|$ 28,300|(3)%|$ 29,292|
|Wearables, Home and Accessories|$ 37,005|(7)%|$ 39,845|(3)%|$ 41,241|
|Services (1)|$ 96,169|13 %|$ 85,200|9 %|$ 78,129|
|Total net sales|$ 391,035|2 %|$ 383,285|(3)%|$ 394,328|

(1) Services net sales include amortization of the deferred value of services bundled in the sales price of certain products.

# iPhone

iPhone net sales were relatively flat during 2024 compared to 2023.

# Mac

Mac net sales increased during 2024 compared to 2023 due primarily to higher net sales of laptops.

# iPad

iPad net sales decreased during 2024 compared to 2023 due primarily to lower net sales of iPad Pro and the entry-level iPad models, partially offset by higher net sales of iPad Air.

# Wearables, Home and Accessories

Wearables, Home and Accessories net sales decreased during 2024 compared to 2023 due primarily to lower net sales of Wearables and Accessories.

# Services

Services net sales increased during 2024 compared to 2023 due primarily to higher net sales from advertising, the App Store® and cloud services.

Apple Inc. | 2024 Form 10-K | 23
# Gross Margin

Products and Services gross margin and gross margin percentage for 2024, 2023 and 2022 were as follows (dollars in millions):

| |2024|2023|2022| | |
|---|---|---|---|---|---|
|Gross margin:|Products|$ 109,633|$ 108,803|$ 114,728| |
| |Services|$ 71,050|$ 60,345|$ 56,054| |
| |Total gross margin|$ 180,683|$ 169,148|$ 170,782| |

| |Gross margin percentage:|2024|2023|2022|
|---|---|---|---|---|
| |Products|37.2%|36.5%|36.3%|
| |Services|73.9%|70.8%|71.7%|
| |Total gross margin percentage|46.2%|44.1%|43.3%|

# Products Gross Margin

Products gross margin and Products gross margin percentage increased during 2024 compared to 2023 due to cost savings, partially offset by a different Products mix and the weakness in foreign currencies relative to the U.S. dollar.

# Services Gross Margin

Services gross margin increased during 2024 compared to 2023 due primarily to higher Services net sales.

Services gross margin percentage increased during 2024 compared to 2023 due to a different Services mix.

The Company’s future gross margins can be impacted by a variety of factors, as discussed in Part I, Item 1A of this Form 10-K under the heading “Risk Factors.” As a result, the Company believes, in general, gross margins will be subject to volatility and downward pressure.

# Operating Expenses

Operating expenses for 2024, 2023 and 2022 were as follows (dollars in millions):

| | |2024|Change|2023|Change|2022|
|---|---|---|---|---|---|---|
|Research and development|$ 31,370| |5 %|$ 29,915|14 %|$ 26,251|
| |Percentage of total net sales|8%| |8%| |7%|
|Selling, general and administrative|$ 26,097| |5 %|$ 24,932|(1)%|$ 25,094|
| |Percentage of total net sales|7%| |7%| |6%|
|Total operating expenses|$ 57,467| |5 %|$ 54,847|7 %|$ 51,345|
| |Percentage of total net sales|15%| |14%| |13%|

# Research and Development

The growth in R&D expense during 2024 compared to 2023 was driven primarily by increases in headcount-related expenses.

# Selling, General and Administrative

Selling, general and administrative expense increased $1.2 billion during 2024 compared to 2023.

Apple Inc. | 2024 Form 10-K | 24
# Provision for Income Taxes

Provision for income taxes, effective tax rate and statutory federal income tax rate for 2024, 2023 and 2022 were as follows (dollars in millions):

| |2024|2023|2022|
|---|---|---|---|
|Provision for income taxes|$ 29,749|$ 16,741|$ 19,300|
|Effective tax rate|24.1%|14.7%|16.2%|
|Statutory federal income tax rate|21%|21%|21%|

The Company’s effective tax rate for 2024 was higher than the statutory federal income tax rate due primarily to a one-time income tax charge of $10.2 billion, net, related to the State Aid Decision (refer to Note 7, “Income Taxes” in the Notes to Consolidated Financial Statements in Part II, Item 8 of this Form 10-K) and state income taxes, partially offset by a lower effective tax rate on foreign earnings, the impact of the U.S. federal R&D credit, and tax benefits from share-based compensation.

The Company’s effective tax rate for 2024 was higher compared to 2023 due primarily to a one-time income tax charge of $10.2 billion, net, related to the State Aid Decision, a higher effective tax rate on foreign earnings and lower tax benefits from share-based compensation.

# Liquidity and Capital Resources

The Company believes its balances of unrestricted cash, cash equivalents and marketable securities, which totaled $140.8 billion as of September 28, 2024, along with cash generated by ongoing operations and continued access to debt markets, will be sufficient to satisfy its cash requirements and capital return program over the next 12 months and beyond.

The Company’s material cash requirements include the following contractual obligations:

# Debt

As of September 28, 2024, the Company had outstanding fixed-rate notes with varying maturities for an aggregate principal amount of $97.3 billion (collectively the “Notes”), with $10.9 billion payable within 12 months. Future interest payments associated with the Notes total $38.5 billion, with $2.6 billion payable within 12 months.

The Company also issues unsecured short-term promissory notes pursuant to a commercial paper program. As of September 28, 2024, the Company had $10.0 billion of commercial paper outstanding, all of which was payable within 12 months.

# Leases

The Company has lease arrangements for certain equipment and facilities, including corporate, data center, manufacturing and retail space. As of September 28, 2024, the Company had fixed lease payment obligations of $15.6 billion, with $2.0 billion payable within 12 months.

# Manufacturing Purchase Obligations

The Company utilizes several outsourcing partners to manufacture subassemblies for the Company’s products and to perform final assembly and testing of finished products. The Company also obtains individual components for its products from a wide variety of individual suppliers. As of September 28, 2024, the Company had manufacturing purchase obligations of $53.0 billion, with $52.9 billion payable within 12 months.

# Other Purchase Obligations

The Company’s other purchase obligations primarily consist of noncancelable obligations to acquire capital assets, including assets related to product manufacturing, and noncancelable obligations related to supplier arrangements, licensed intellectual property and content, and distribution rights. As of September 28, 2024, the Company had other purchase obligations of $12.0 billion, with $4.1 billion payable within 12 months.

# Deemed Repatriation Tax Payable

As of September 28, 2024, the balance of the deemed repatriation tax payable imposed by the U.S. Tax Cuts and Jobs Act of 2017 (the “TCJA”) was $16.5 billion, with $7.2 billion expected to be paid within 12 months.

Apple Inc. | 2024 Form 10-K | 25
# State Aid Decision Tax Payable

As of September 28, 2024, the Company had an obligation to pay €14.2 billion or $15.8 billion to Ireland in connection with the State Aid Decision, all of which was expected to be paid within 12 months. The funds necessary to settle the obligation were held in escrow as of September 28, 2024, and restricted from general use.

# Capital Return Program

In addition to its contractual cash requirements, the Company has an authorized share repurchase program. The program does not obligate the Company to acquire a minimum amount of shares. As of September 28, 2024, the Company’s quarterly cash dividend was $0.25 per share. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board.

In May 2024, the Company announced a new share repurchase program of up to $110 billion and raised its quarterly dividend from $0.24 to $0.25 per share beginning in May 2024. During 2024, the Company repurchased $95.0 billion of its common stock and paid dividends and dividend equivalents of $15.2 billion.

# Recent Accounting Pronouncements

# Income Taxes

In December 2023, the Financial Accounting Standards Board (the “FASB”) issued Accounting Standards Update (“ASU”) No. 2023-09, Income Taxes (Topic 740): Improvements to Income Tax Disclosures (“ASU 2023-09”), which will require the Company to disclose specified additional information in its income tax rate reconciliation and provide additional information for reconciling items that meet a quantitative threshold. ASU 2023-09 will also require the Company to disaggregate its income taxes paid disclosure by federal, state and foreign taxes, with further disaggregation required for significant individual jurisdictions. The Company will adopt ASU 2023-09 in its fourth quarter of 2026 using a prospective transition method.

# Segment Reporting

In November 2023, the FASB issued ASU No. 2023-07, Segment Reporting (Topic 280): Improvements to Reportable Segment Disclosures (“ASU 2023-07”), which will require the Company to disclose segment expenses that are significant and regularly provided to the Company’s chief operating decision maker (“CODM”). In addition, ASU 2023-07 will require the Company to disclose the title and position of its CODM and how the CODM uses segment profit or loss information in assessing segment performance and deciding how to allocate resources. The Company will adopt ASU 2023-07 in its fourth quarter of 2025 using a retrospective transition method.

# Critical Accounting Estimates

The preparation of financial statements and related disclosures in conformity with U.S. generally accepted accounting principles (“GAAP”) and the Company’s discussion and analysis of its financial condition and operating results require the Company’s management to make judgments, assumptions and estimates that affect the amounts reported. Note 1, “Summary of Significant Accounting Policies” of the Notes to Consolidated Financial Statements in Part II, Item 8 of this Form 10-K describes the significant accounting policies and methods used in the preparation of the Company’s consolidated financial statements. Management bases its estimates on historical experience and on various other assumptions it believes to be reasonable under the circumstances, the results of which form the basis for
# Item 7A. Quantitative and Qualitative Disclosures About Market Risk

The Company is exposed to economic risk from interest rates and foreign exchange rates. The Company uses various strategies to manage these risks; however, they may still impact the Company’s consolidated financial statements.

# Interest Rate Risk

The Company is primarily exposed to fluctuations in U.S. interest rates and their impact on the Company’s investment portfolio and term debt. Increases in interest rates will negatively affect the fair value of the Company’s investment portfolio and increase the interest expense on the Company’s term debt. To protect against interest rate risk, the Company may use derivative instruments, offset interest rate–sensitive assets and liabilities, or control duration of the investment and term debt portfolios.

|Interest Rate Sensitive Instrument|Hypothetical Interest Rate Increase|Potential Impact|2024|2023|
|---|---|---|---|---|
|Investment portfolio|100 basis points, all tenors|Decline in fair value|$ 2,755|$ 3,089|
|Term debt|100 basis points, all tenors|Increase in annual interest expense|$ 139|$ 194|

# Foreign Exchange Rate Risk

The Company’s exposure to foreign exchange rate risk relates primarily to the Company being a net receiver of currencies other than the U.S. dollar. Changes in exchange rates, and in particular a strengthening of the U.S. dollar, will negatively affect the Company’s net sales and gross margins as expressed in U.S. dollars. Fluctuations in exchange rates may also affect the fair values of certain of the Company’s assets and liabilities. To protect against foreign exchange rate risk, the Company may use derivative instruments, offset exposures, or adjust local currency pricing of its products and services. However, the Company may choose to not hedge certain foreign currency exposures for a variety of reasons, including accounting considerations or prohibitive cost.

The Company applied a value-at-risk (“VAR”) model to its foreign currency derivative positions to assess the potential impact of fluctuations in exchange rates. The VAR model used a Monte Carlo simulation. The VAR is the maximum expected loss in fair value, for a given confidence interval, to the Company’s foreign currency derivative positions due to adverse movements in rates. Based on the results of the model, the Company estimates, with 95% confidence, a maximum one-day loss in fair value of $538 million and $669 million as of September 28, 2024 and September 30, 2023, respectively. Changes in the Company’s underlying foreign currency exposures, which were excluded from the assessment, generally offset changes in the fair values of the Company’s foreign currency derivatives.

Apple Inc. | 2024 Form 10-K | 27"""

questions = [
    "What were Apple Inc.'s share repurchase and dividend payments in 2024?",
    "What does the result of the VAR model indicate about the company's estimated maximum one-day loss in fair value?",
    "How does the company address the changing demands of customers and users?",
    "What impact do international operations have on the company's financials?",
    "What is the company's approach to attracting and retaining qualified employees?"

] # Example questions

# === LOAD MODEL AND TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True) # or remove load_in_4bit
model = PeftModel.from_pretrained(base_model, model_path, device_map="auto")
model.eval()

# === INFERENCE FUNCTION ===
def generate_response(context, question):
    prompt = f"Context: {context}\nPrompt: {question}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # or "cpu"
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Response:")[1].strip()
    return response

# === GENERATE RESPONSES ===
for question in questions:
    generated_response = generate_response(raw_text, question)
    print("Question:", question)
    print("Generated Response:", generated_response)
    print("-" * 50)


# === (Optional) Save to JSON (adapt as needed) ===
results = []
for question in questions:
    generated_response = generate_response(raw_text, question)
    results.append({"question": question, "generated_response": generated_response})

import json
with open("generated_responses_from_text.json", "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, indent=4, ensure_ascii=False)
