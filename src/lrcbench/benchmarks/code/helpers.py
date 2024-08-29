import numpy as np
import pandas as pd

SIZE = 50
np.random.seed(42)


def uniform_sample(start, end, size=1):
    return np.random.uniform(start, end, size)


def uniform_int_sample(start, end, size=1):
    return np.random.randint(start, end + 1, size)


def uniform_choice(options, size=1):
    return np.random.choice(options, size)


ranges = {
    "funding_amount": (0, 15_000_000_000),
    "employee_count": (0, 5_000_000),
    "years_since_founding": (0, 50),
    "revenue": (0, 5_000_000_000),
    "burn_rate": (0, 1_000_000_000),
    "customer_count": (0, 1_000_000_000),
    "monthly_active_users": (0, 1_000_000_000),
    "churn_rate": (0, 1),
    "growth_rate": (-1, 10),
    "valuation": (0, 1_000_000_000_000),
    "industry": ["Technology", "Healthcare", "Finance", "Retail", "Education"],
    "business_model": ["B2B", "B2C", "Marketplace", "SaaS", "Hardware"],
    "target_market": ["Small Business", "Enterprise", "Consumer", "Government"],
    "location": ["US", "Europe", "Asia", "Africa", "South America"],
    "founder_experience": ["First-time", "Serial", "Industry Expert"],
    "product_stage": ["Idea", "MVP", "Beta", "Live", "Growth", "Mature"],
    "funding_round": ["Seed", "Series A", "Series B", "Series C", "Series D+"],
    "has_patents": [True, False],
    "is_profitable": [True, False],
    "has_partnerships": [True, False],
}

BASE_DATAFRAME = pd.DataFrame(
    {
        "funding_amount": uniform_sample(*ranges["funding_amount"], SIZE),
        "employee_count": uniform_int_sample(*ranges["employee_count"], SIZE),
        "years_since_founding": uniform_int_sample(
            *ranges["years_since_founding"], SIZE
        ),
        "revenue": uniform_sample(*ranges["revenue"], SIZE),
        "burn_rate": uniform_sample(*ranges["burn_rate"], SIZE),
        "customer_count": uniform_int_sample(*ranges["customer_count"], SIZE),
        "monthly_active_users": uniform_int_sample(
            *ranges["monthly_active_users"], SIZE
        ),
        "churn_rate": uniform_sample(*ranges["churn_rate"], SIZE),
        "growth_rate": uniform_sample(*ranges["growth_rate"], SIZE),
        "valuation": uniform_sample(*ranges["valuation"], SIZE),
        "industry": uniform_choice(ranges["industry"], SIZE),
        "business_model": uniform_choice(ranges["business_model"], SIZE),
        "target_market": uniform_choice(ranges["target_market"], SIZE),
        "location": uniform_choice(ranges["location"], SIZE),
        "founder_experience": uniform_choice(ranges["founder_experience"], SIZE),
        "product_stage": uniform_choice(ranges["product_stage"], SIZE),
        "funding_round": uniform_choice(ranges["funding_round"], SIZE),
        "has_patents": uniform_choice(ranges["has_patents"], SIZE),
        "is_profitable": uniform_choice(ranges["is_profitable"], SIZE),
        "has_partnerships": uniform_choice(ranges["has_partnerships"], SIZE),
    }
)


# Helpers that each apply a single reasonable transformation to the data


def remove_funding_outliers(df):
    funding_threshold = df["funding_amount"].quantile(0.99)
    return df[df["funding_amount"] <= funding_threshold]


def cap_employee_count(df):
    max_employees = 100000
    return df.assign(employee_count=df["employee_count"].clip(upper=max_employees))


def normalize_revenue(df):
    return df.assign(revenue_per_employee=df["revenue"] / df["employee_count"])


def categorize_company_age(df):
    bins = [0, 2, 5, 10, float("inf")]
    labels = ["Startup", "Early Stage", "Growth Stage", "Mature"]
    return df.assign(
        company_age_category=pd.cut(
            df["years_since_founding"], bins=bins, labels=labels, right=False
        )
    )


def calculate_runway(df):
    return df.assign(
        runway_months=(df["funding_amount"] / df["burn_rate"]).clip(lower=0)
    )


def segment_customer_base(df):
    bins = [0, 100, 1000, 10000, float("inf")]
    labels = ["Small", "Medium", "Large", "Enterprise"]
    return df.assign(
        customer_segment=pd.cut(
            df["customer_count"], bins=bins, labels=labels, right=False
        )
    )


def compute_user_engagement(df):
    return df.assign(engagement_ratio=df["monthly_active_users"] / df["customer_count"])


def adjust_growth_rate(df):
    return df.assign(adjusted_growth_rate=df["growth_rate"].clip(lower=-1, upper=5))


def calculate_valuation_multiple(df):
    return df.assign(
        valuation_to_revenue_multiple=df["valuation"] / df["revenue"].replace(0, 1)
    )


def encode_categorical_features(df):
    categorical_columns = [
        "industry",
        "business_model",
        "target_market",
        "location",
        "founder_experience",
        "product_stage",
        "funding_round",
    ]
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def create_composite_score(df):
    df["composite_score"] = (
        df["funding_amount"].rank(pct=True)
        + df["employee_count"].rank(pct=True)
        + df["revenue"].rank(pct=True)
        + df["growth_rate"].rank(pct=True)
        + df["valuation"].rank(pct=True)
    ) / 5
    return df


def calculate_customer_acquisition_cost(df):
    marketing_spend = df["burn_rate"] * 0.3
    new_customers = df["customer_count"] * df["growth_rate"]
    return df.assign(cac=marketing_spend / new_customers)


def identify_unicorns(df):
    return df.assign(is_unicorn=df["valuation"] >= 1_000_000_000)


def categorize_burn_rate(df):
    return df.assign(
        burn_rate_category=pd.cut(
            df["burn_rate"],
            bins=[0, 100000, 1000000, float("inf")],
            labels=["Low", "Medium", "High"],
        )
    )


def calculate_efficiency_score(df):
    return df.assign(
        efficiency_score=(df["revenue"] / df["burn_rate"])
        * (1 + df["growth_rate"])
        * (1 - df["churn_rate"])
    )


def identify_market_leaders(df):
    top_5_percent = df["market_share"].quantile(0.95)
    return df.assign(is_market_leader=df["market_share"] >= top_5_percent)


def calculate_runway_months(df):
    return df.assign(runway_months=df["funding_amount"] / (df["burn_rate"] + 1e-6))


def flag_high_churn(df):
    high_churn_threshold = df["churn_rate"].quantile(0.75)
    return df.assign(high_churn_flag=df["churn_rate"] > high_churn_threshold)


def calculate_revenue_multiple(df):
    return df.assign(revenue_multiple=df["valuation"] / (df["revenue"] + 1))


def categorize_growth_stage(df):
    conditions = [
        (df["revenue"] < 1_000_000) & (df["years_since_founding"] <= 2),
        (df["revenue"] < 10_000_000) & (df["years_since_founding"] <= 5),
        (df["revenue"] < 100_000_000),
        (df["revenue"] >= 100_000_000),
    ]
    choices = ["Seed", "Early", "Growth", "Late"]
    return df.assign(growth_stage=np.select(conditions, choices, default="Unknown"))


def calculate_net_burn(df):
    return df.assign(net_burn=df["burn_rate"] - df["revenue"])


def identify_capital_efficient(df):
    efficiency_ratio = (df["revenue"] * df["growth_rate"]) / df["funding_amount"]
    top_quartile = efficiency_ratio.quantile(0.75)
    return df.assign(is_capital_efficient=efficiency_ratio > top_quartile)


def calculate_ltv_to_cac_ratio(df):
    ltv = df["revenue_per_user"] / df["churn_rate"]
    cac = df["burn_rate"] / (df["customer_count"] * df["growth_rate"])
    return df.assign(ltv_to_cac_ratio=ltv / cac)


def flag_potential_acquirers(df):
    conditions = (df["cash_reserves"] > df["market_cap"]) & (
        df["is_profitable"] == True
    )
    return df.assign(potential_acquirer=conditions)


def calculate_rule_of_40(df):
    return df.assign(rule_of_40_score=df["growth_rate"] + (df["profit_margin"] * 100))


def identify_hypergrowth(df):
    return df.assign(is_hypergrowth=df["growth_rate"] > 1)  # Over 100% growth


def calculate_magic_number(df):
    new_arr = df["revenue"] * df["growth_rate"]
    sales_marketing_spend = df["burn_rate"] * 0.4  # Assuming 40% of burn is S&M
    return df.assign(magic_number=new_arr / sales_marketing_spend)


def categorize_product_market_fit(df):
    conditions = [
        df["churn_rate"] < 0.02,
        (df["churn_rate"] >= 0.02) & (df["churn_rate"] < 0.05),
        df["churn_rate"] >= 0.05,
    ]
    choices = ["Strong", "Moderate", "Weak"]
    return df.assign(
        product_market_fit=np.select(conditions, choices, default="Unknown")
    )


def calculate_revenue_concentration_risk(df):
    herfindahl_index = (df["revenue"] / df["revenue"].sum()) ** 2
    return df.assign(revenue_concentration_risk=herfindahl_index.sum())


def identify_cash_flow_positive(df):
    return df.assign(is_cash_flow_positive=df["revenue"] > df["burn_rate"])


def filter_high_growth_companies(df):
    return df[df["growth_rate"] > 0.5]


def sort_by_efficiency(df):
    return df.sort_values(
        by="revenue", key=lambda x: x / df["employee_count"], ascending=False
    )


def sample_top_performers(df):
    return df.nlargest(10, "valuation")


def convert_to_millions(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return df.assign(**{col: df[col] / 1e6 for col in numeric_columns})


def pivot_by_industry(df):
    return df.pivot_table(values="revenue", index="industry", aggfunc="sum")


def rank_by_funding(df):
    return df.assign(
        funding_rank=df["funding_amount"].rank(method="dense", ascending=False)
    )


def filter_recent_startups(df):
    return df[df["years_since_founding"] <= 5]


def group_by_location(df):
    return df.groupby("location").agg({"revenue": "sum", "employee_count": "mean"})


def calculate_percentiles(df):
    return df.rank(pct=True)


def filter_profitable_companies(df):
    return df[df["is_profitable"] == True]


def sort_by_customer_count(df):
    return df.sort_values("customer_count", ascending=False)


def sample_random_companies(df):
    return df.sample(n=min(20, len(df)))


def filter_by_business_model(df, model="SaaS"):
    return df[df["business_model"] == model]


def calculate_moving_average(df):
    return df.sort_values("years_since_founding").rolling(window=3).mean()


def normalize_features(df):
    return (df - df.mean()) / df.std()


def bin_company_sizes(df):
    return pd.cut(
        df["employee_count"],
        bins=[0, 10, 50, 250, 1000, float("inf")],
        labels=["Micro", "Small", "Medium", "Large", "Enterprise"],
    )


def filter_by_valuation_range(df, min_val=1e6, max_val=1e9):
    return df[(df["valuation"] >= min_val) & (df["valuation"] <= max_val)]


def calculate_correlation_matrix(df):
    return df.corr()


def group_by_founder_experience(df):
    return df.groupby("founder_experience").agg(
        {"funding_amount": "mean", "valuation": "median"}
    )


def filter_companies_with_patents(df):
    return df[df["has_patents"] == True]


def calculate_market_penetration(df):
    total_addressable_market = 1e9
    return df.assign(market_penetration=df["customer_count"] / total_addressable_market)


def identify_moonshot_projects(df):
    return df.assign(
        is_moonshot=(df["burn_rate"] > 10 * df["revenue"]) & (df["growth_rate"] > 2)
    )


def calculate_founder_equity_dilution(df):
    initial_equity = 1
    rounds = {
        "Seed": 0.2,
        "Series A": 0.15,
        "Series B": 0.1,
        "Series C": 0.08,
        "Series D+": 0.05,
    }
    df["founder_equity"] = (
        initial_equity - df["funding_round"].map(rounds).fillna(0).cumsum()
    )
    return df


def estimate_time_to_profitability(df):
    monthly_loss = df["burn_rate"] - df["revenue"]
    monthly_improvement = df["revenue"] * df["growth_rate"] / 12
    return df.assign(months_to_profitability=monthly_loss / monthly_improvement)


def calculate_innovation_index(df):
    patent_factor = df["has_patents"].map({True: 1.5, False: 1})
    return df.assign(
        innovation_index=(df["funding_amount"] / df["years_since_founding"])
        * patent_factor
    )


def identify_potential_acquisitions(df):
    is_potential_target = (df["cash_reserves"] < df["burn_rate"] * 6) & (
        df["growth_rate"] > 0.5
    )
    return df.assign(acquisition_target=is_potential_target)


def calculate_talent_density(df):
    return df.assign(talent_density=df["revenue"] / df["employee_count"])


def estimate_market_share(df):
    total_market_size = df.groupby("industry")["revenue"].transform("sum")
    return df.assign(estimated_market_share=df["revenue"] / total_market_size)


def calculate_customer_lifetime_value(df):
    average_revenue_per_user = df["revenue"] / df["customer_count"]
    customer_lifespan = 1 / df["churn_rate"]
    return df.assign(customer_ltv=average_revenue_per_user * customer_lifespan)


def identify_industry_disruptors(df):
    industry_avg_growth = df.groupby("industry")["growth_rate"].transform("mean")
    return df.assign(is_disruptor=df["growth_rate"] > 2 * industry_avg_growth)


def calculate_burn_multiple(df):
    net_new_arr = df["revenue"] * df["growth_rate"]
    return df.assign(burn_multiple=df["burn_rate"] / net_new_arr)


def estimate_total_addressable_market(df):
    market_sizes = {
        "Technology": 1e12,
        "Healthcare": 8e11,
        "Finance": 1.5e12,
        "Retail": 5e11,
        "Education": 2e11,
    }
    return df.assign(estimated_tam=df["industry"].map(market_sizes))


def calculate_revenue_per_founder(df):
    avg_founders = {"First-time": 2, "Serial": 1.5, "Industry Expert": 1.8}
    estimated_founders = df["founder_experience"].map(avg_founders)
    return df.assign(revenue_per_founder=df["revenue"] / estimated_founders)


def identify_cash_efficient_growth(df):
    efficiency_ratio = (df["revenue"] * df["growth_rate"]) / df["funding_amount"]
    return df.assign(
        is_cash_efficient_growth=efficiency_ratio > efficiency_ratio.median()
    )


def calculate_product_adoption_rate(df):
    total_addressable_users = 1e8
    return df.assign(
        product_adoption_rate=df["monthly_active_users"] / total_addressable_users
    )


def estimate_viral_coefficient(df):
    viral_factor = (df["monthly_active_users"] / df["customer_count"]).clip(upper=10)
    return df.assign(estimated_viral_coefficient=viral_factor * df["growth_rate"])


def calculate_customer_acquisition_efficiency(df):
    cac = df["burn_rate"] * 0.4 / (df["customer_count"] * df["growth_rate"])
    ltv = df["revenue_per_user"] / df["churn_rate"]
    return df.assign(cac_efficiency=ltv / cac)


def identify_potential_ipo_candidates(df):
    ipo_conditions = (
        (df["revenue"] > 1e8)
        & (df["growth_rate"] > 0.3)
        & (df["years_since_founding"] > 7)
    )
    return df.assign(ipo_candidate=ipo_conditions)


def calculate_research_and_development_intensity(df):
    rd_spend = df["burn_rate"] * 0.3
    return df.assign(rd_intensity=rd_spend / df["revenue"])


def estimate_customer_satisfaction_score(df):
    base_score = 70
    churn_impact = (0.05 - df["churn_rate"]) * 200
    growth_impact = df["growth_rate"] * 50
    return df.assign(estimated_csat=base_score + churn_impact + growth_impact)


def calculate_operational_efficiency(df):
    return df.assign(
        operational_efficiency=df["revenue"]
        / (df["burn_rate"] + df["employee_count"] * 100000)
    )


def identify_sustainable_growth_companies(df):
    sustainable_growth = (
        (df["growth_rate"] > 0.2)
        & (df["burn_rate"] < df["revenue"])
        & (df["churn_rate"] < 0.1)
    )
    return df.assign(is_sustainable_growth=sustainable_growth)


def calculate_product_market_fit_score(df):
    engagement_score = df["monthly_active_users"] / df["customer_count"]
    retention_score = 1 - df["churn_rate"]
    growth_score = df["growth_rate"].clip(lower=0, upper=1)
    return df.assign(
        product_market_fit_score=(engagement_score + retention_score + growth_score) / 3
    )


def estimate_brand_value(df):
    base_value = df["valuation"] * 0.1
    growth_factor = 1 + df["growth_rate"]
    market_leader_bonus = df["is_market_leader"].map({True: 1.5, False: 1})
    return df.assign(
        estimated_brand_value=base_value * growth_factor * market_leader_bonus
    )


def calculate_employee_productivity(df):
    return df.assign(productivity_per_employee=df["revenue"] / df["employee_count"])


def identify_pivot_potential(df):
    pivot_score = (1 - df["product_market_fit_score"]) * df["runway_months"] / 12
    return df.assign(pivot_potential=pivot_score)


def estimate_customer_acquisition_channels(df):
    channel_distribution = {
        "Organic": 0.3 + (1 - df["burn_rate"] / df["revenue"]) * 0.2,
        "Paid": 0.2 + (df["burn_rate"] / df["revenue"]) * 0.3,
        "Partnerships": 0.15 + df["has_partnerships"].map({True: 0.1, False: 0}),
        "Referral": 0.1 + df["growth_rate"] * 0.2,
        "Other": 0.05,
    }
    return df.assign(**channel_distribution)


def calculate_network_effects_score(df):
    base_score = df["monthly_active_users"] / df["customer_count"]
    growth_multiplier = 1 + df["growth_rate"]
    business_model_factor = df["business_model"].map(
        {"Marketplace": 1.5, "SaaS": 1.2, "B2B": 1.1, "B2C": 1.3, "Hardware": 1}
    )
    return df.assign(
        network_effects_score=base_score * growth_multiplier * business_model_factor
    )


def estimate_technology_stack_complexity(df):
    base_complexity = df["employee_count"] * 0.01
    scaling_factor = np.log1p(df["monthly_active_users"])
    product_stage_factor = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 1, "Beta": 1.5, "Live": 2, "Growth": 2.5, "Mature": 3}
    )
    return df.assign(
        tech_stack_complexity=base_complexity * scaling_factor * product_stage_factor
    )


def calculate_investor_roi_potential(df):
    expected_exit_multiple = 5
    time_to_exit = 7
    annual_roi = (df["valuation"] * expected_exit_multiple / df["funding_amount"]) ** (
        1 / time_to_exit
    ) - 1
    return df.assign(investor_roi_potential=annual_roi)


def identify_regulatory_risk(df):
    industry_risk = df["industry"].map(
        {
            "Technology": 0.3,
            "Healthcare": 0.8,
            "Finance": 0.9,
            "Retail": 0.4,
            "Education": 0.6,
        }
    )
    size_risk = df["valuation"].clip(upper=1e9) / 1e9
    return df.assign(regulatory_risk_score=industry_risk * size_risk)


def calculate_customer_concentration_risk(df):
    herfindahl_index = (df["revenue"] / df["revenue"].sum()) ** 2
    return df.assign(customer_concentration_risk=herfindahl_index.sum())


def estimate_talent_turnover_rate(df):
    base_rate = 0.15
    growth_factor = 1 + df["growth_rate"] * 0.5
    burn_factor = 1 + (df["burn_rate"] / df["funding_amount"]) * 0.3
    return df.assign(estimated_talent_turnover=base_rate * growth_factor * burn_factor)


def calculate_product_complexity_score(df):
    base_complexity = df["employee_count"] * 0.05
    tech_factor = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.2,
            "Retail": 1,
            "Education": 1.1,
        }
    )
    stage_factor = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 1, "Beta": 1.5, "Live": 2, "Growth": 2.5, "Mature": 3}
    )
    return df.assign(
        product_complexity_score=base_complexity * tech_factor * stage_factor
    )


def identify_potential_strategic_partnerships(df):
    partnership_score = (
        df["market_share"] * df["growth_rate"] * df["innovation_index"]
    ).rank(pct=True)
    return df.assign(strategic_partnership_potential=partnership_score)


def calculate_founder_commitment_score(df):
    equity_factor = 1 - df["funding_round"].map(
        {
            "Seed": 0.1,
            "Series A": 0.2,
            "Series B": 0.3,
            "Series C": 0.4,
            "Series D+": 0.5,
        }
    )
    experience_factor = df["founder_experience"].map(
        {"First-time": 1.2, "Serial": 1, "Industry Expert": 1.1}
    )
    return df.assign(founder_commitment_score=equity_factor * experience_factor)


def estimate_product_development_velocity(df):
    base_velocity = df["employee_count"] * 0.1
    funding_factor = np.log1p(df["funding_amount"]) / 10
    stage_factor = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 1, "Beta": 1.5, "Live": 2, "Growth": 1.8, "Mature": 1.5}
    )
    return df.assign(
        product_development_velocity=base_velocity * funding_factor * stage_factor
    )


def calculate_market_opportunity_score(df):
    tam_factor = np.log1p(df["estimated_tam"]) / 10
    growth_factor = 1 + df["growth_rate"]
    competition_factor = 1 - df["estimated_market_share"]
    return df.assign(
        market_opportunity_score=tam_factor * growth_factor * competition_factor
    )


def identify_potential_talent_magnets(df):
    talent_attraction_score = (
        df["growth_rate"] + df["innovation_index"] + df["funding_amount"].rank(pct=True)
    ) / 3
    return df.assign(talent_magnet_score=talent_attraction_score)


def calculate_global_expansion_readiness(df):
    base_readiness = df["years_since_founding"] * 0.1
    funding_factor = np.log1p(df["funding_amount"]) / 10
    market_factor = df["estimated_market_share"].clip(upper=0.1) * 10
    return df.assign(
        global_expansion_readiness=base_readiness * funding_factor * market_factor
    )


def estimate_customer_support_efficiency(df):
    support_staff_ratio = 0.1
    tickets_per_customer = 0.5
    tickets_per_staff = 100
    estimated_support_staff = df["employee_count"] * support_staff_ratio
    estimated_tickets = df["customer_count"] * tickets_per_customer
    return df.assign(
        support_efficiency=estimated_tickets
        / (estimated_support_staff * tickets_per_staff)
    )


def calculate_product_stickiness_score(df):
    usage_factor = df["monthly_active_users"] / df["customer_count"]
    retention_factor = 1 - df["churn_rate"]
    growth_factor = 1 + df["growth_rate"]
    return df.assign(product_stickiness=usage_factor * retention_factor * growth_factor)


def identify_potential_market_consolidators(df):
    consolidation_score = (
        df["market_share"]
        + df["cash_reserves"].rank(pct=True)
        + df["operational_efficiency"]
    ) / 3
    return df.assign(market_consolidator_potential=consolidation_score)


def calculate_innovation_to_execution_ratio(df):
    innovation_score = (
        df["has_patents"].map({True: 1, False: 0}) + df["product_complexity_score"]
    )
    execution_score = df["operational_efficiency"] + df["growth_rate"]
    return df.assign(innovation_execution_ratio=innovation_score / execution_score)


def estimate_customer_feedback_sentiment(df):
    base_sentiment = 0.5
    product_fit_impact = df["product_market_fit_score"] * 0.3
    growth_impact = df["growth_rate"].clip(lower=-0.5, upper=0.5) * 0.2
    return df.assign(
        estimated_customer_sentiment=base_sentiment + product_fit_impact + growth_impact
    )


def calculate_ecosystem_impact_score(df):
    market_impact = df["estimated_market_share"] * 5
    employment_impact = df["employee_count"] / 1000
    innovation_impact = df["has_patents"].map({True: 1, False: 0}) * df[
        "funding_amount"
    ].rank(pct=True)
    return df.assign(
        ecosystem_impact_score=(market_impact + employment_impact + innovation_impact)
        / 3
    )


def identify_potential_category_creators(df):
    category_creation_score = (
        df["innovation_index"]
        + df["market_opportunity_score"]
        + df["funding_amount"].rank(pct=True)
    ) / 3
    return df.assign(
        category_creator_potential=category_creation_score
        > category_creation_score.quantile(0.9)
    )


def calculate_pricing_power_index(df):
    market_share_factor = df["estimated_market_share"].clip(upper=0.5) * 2
    product_uniqueness = (
        df["has_patents"].map({True: 1, False: 0}) + df["product_complexity_score"]
    )
    return df.assign(pricing_power_index=market_share_factor * product_uniqueness)


def estimate_remote_work_adaptability(df):
    tech_factor = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 0.8,
            "Finance": 1.2,
            "Retail": 0.7,
            "Education": 1.3,
        }
    )
    size_factor = 1 - (df["employee_count"] / 1000).clip(upper=0.5)
    product_factor = df["product_stage"].map(
        {"Idea": 1.5, "MVP": 1.3, "Beta": 1.2, "Live": 1, "Growth": 0.9, "Mature": 0.8}
    )
    return df.assign(
        remote_work_adaptability=tech_factor * size_factor * product_factor
    )


def calculate_intellectual_property_strength(df):
    patent_score = df["has_patents"].map({True: 1, False: 0})
    rd_intensity = df["burn_rate"] * 0.3 / df["revenue"]
    funding_factor = np.log1p(df["funding_amount"]) / 10
    return df.assign(ip_strength=patent_score * (1 + rd_intensity) * funding_factor)


def identify_potential_unicorn_trajectories(df):
    growth_trajectory = df["growth_rate"] ** 2 * df["years_since_founding"]
    funding_trajectory = np.log1p(df["funding_amount"]) / np.log1p(
        df["years_since_founding"]
    )
    market_trajectory = df["market_opportunity_score"] * df["estimated_market_share"]
    unicorn_score = (growth_trajectory + funding_trajectory + market_trajectory) / 3
    return df.assign(unicorn_trajectory_score=unicorn_score)


def calculate_customer_success_index(df):
    retention_score = 1 - df["churn_rate"]
    expansion_score = df["growth_rate"].clip(lower=0)
    support_score = df["support_efficiency"]
    return df.assign(
        customer_success_index=(retention_score + expansion_score + support_score) / 3
    )


def estimate_product_launch_frequency(df):
    base_frequency = 2  # launches per year
    stage_factor = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 1, "Beta": 1.5, "Live": 2, "Growth": 1.5, "Mature": 1}
    )
    team_size_factor = np.log1p(df["employee_count"]) / 5
    return df.assign(
        estimated_launch_frequency=base_frequency * stage_factor * team_size_factor
    )


def calculate_founder_market_fit_score(df):
    experience_score = df["founder_experience"].map(
        {"First-time": 0.7, "Serial": 1, "Industry Expert": 1.2}
    )
    market_growth = df["growth_rate"].clip(lower=0)
    product_fit = df["product_market_fit_score"]
    return df.assign(
        founder_market_fit=experience_score * (1 + market_growth) * product_fit
    )


def identify_potential_acquisition_targets(df):
    acquisition_attractiveness = (
        df["innovation_index"] + df["market_share"].rank(pct=True) + df["growth_rate"]
    ) / 3
    financial_distress = (df["burn_rate"] / df["funding_amount"]).clip(upper=1)
    return df.assign(
        acquisition_target_score=acquisition_attractiveness * (1 - financial_distress)
    )


def calculate_product_evolution_rate(df):
    base_rate = 1
    funding_factor = np.log1p(df["funding_amount"]) / 10
    team_factor = np.log1p(df["employee_count"]) / 5
    market_pressure = 1 + df["growth_rate"]
    return df.assign(
        product_evolution_rate=base_rate
        * funding_factor
        * team_factor
        * market_pressure
    )


def estimate_community_engagement_level(df):
    base_engagement = df["monthly_active_users"] / df["customer_count"]
    product_factor = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 0.7, "Beta": 1, "Live": 1.2, "Growth": 1.5, "Mature": 1.3}
    )
    viral_factor = df["estimated_viral_coefficient"]
    return df.assign(
        community_engagement_level=base_engagement * product_factor * viral_factor
    )


def calculate_talent_attraction_power(df):
    growth_appeal = 1 + df["growth_rate"]
    funding_appeal = np.log1p(df["funding_amount"]) / 10
    innovation_appeal = df["innovation_index"]
    market_appeal = df["market_opportunity_score"]
    return df.assign(
        talent_attraction_power=(
            growth_appeal + funding_appeal + innovation_appeal + market_appeal
        )
        / 4
    )


def identify_potential_industry_disruptors(df):
    disruption_score = (
        df["innovation_index"] + df["growth_rate"] + df["market_opportunity_score"]
    ) / 3
    industry_avg_disruption = df.groupby("industry")["disruption_score"].transform(
        "mean"
    )
    return df.assign(
        disruptor_potential=df["disruption_score"] > 2 * industry_avg_disruption
    )


def calculate_product_market_expansion_potential(df):
    current_market_size = df["estimated_tam"] * df["estimated_market_share"]
    growth_potential = df["growth_rate"].clip(lower=0)
    adjacent_market_factor = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.2,
            "Retail": 1.4,
            "Education": 1.1,
        }
    )
    return df.assign(
        market_expansion_potential=np.log1p(current_market_size)
        * (1 + growth_potential)
        * adjacent_market_factor
    )


def estimate_customer_education_need(df):
    product_complexity = df["product_complexity_score"]
    market_maturity = df["industry"].map(
        {
            "Technology": 0.8,
            "Healthcare": 1,
            "Finance": 0.9,
            "Retail": 0.7,
            "Education": 1.1,
        }
    )
    adoption_rate = df["product_adoption_rate"]
    return df.assign(
        customer_education_need=product_complexity * market_maturity / adoption_rate
    )


def calculate_startup_runway_stress(df):
    burn_rate_stress = df["burn_rate"] / df["funding_amount"]
    growth_pressure = 1 + df["growth_rate"]
    market_competition = 1 / (df["estimated_market_share"] + 0.01)
    return df.assign(
        runway_stress_score=burn_rate_stress * growth_pressure * market_competition
    )


def identify_potential_pivot_candidates(df):
    pivot_pressure = (1 - df["product_market_fit_score"]) * (
        1 - df["estimated_market_share"]
    )
    financial_flexibility = df["runway_months"] / 12
    team_adaptability = 1 / (df["years_since_founding"] + 1)
    return df.assign(
        pivot_candidate_score=pivot_pressure * financial_flexibility * team_adaptability
    )


def calculate_customer_segment_diversification(df):
    segment_distribution = {
        "Enterprise": 0.4 * df["customer_count"],
        "Mid-market": 0.3 * df["customer_count"],
        "SMB": 0.2 * df["customer_count"],
        "Consumer": 0.1 * df["customer_count"],
    }
    segment_shares = pd.DataFrame(segment_distribution, index=df.index)
    herfindahl_index = (segment_shares**2).sum(axis=1) / (df["customer_count"] ** 2)
    return df.assign(customer_segment_diversity=1 - herfindahl_index)


def estimate_product_virality_potential(df):
    base_virality = df["estimated_viral_coefficient"]
    social_factor = df["industry"].map(
        {
            "Technology": 1.3,
            "Healthcare": 0.9,
            "Finance": 0.8,
            "Retail": 1.1,
            "Education": 1.2,
        }
    )
    user_base_factor = np.log1p(df["monthly_active_users"]) / 10
    return df.assign(
        virality_potential=base_virality * social_factor * user_base_factor
    )


def calculate_regulatory_compliance_burden(df):
    industry_burden = df["industry"].map(
        {
            "Technology": 0.7,
            "Healthcare": 1.5,
            "Finance": 1.8,
            "Retail": 0.9,
            "Education": 1.1,
        }
    )
    size_burden = np.log1p(df["employee_count"]) / 5
    international_factor = df["location"].map(
        {"US": 1, "Europe": 1.2, "Asia": 1.1, "Africa": 0.9, "South America": 1}
    )
    return df.assign(
        regulatory_burden=industry_burden * size_burden * international_factor
    )


def identify_potential_breakout_growth_companies(df):
    growth_acceleration = df["growth_rate"] / df["years_since_founding"]
    market_opportunity = df["market_opportunity_score"]
    execution_capability = df["operational_efficiency"]
    breakout_score = (
        growth_acceleration + market_opportunity + execution_capability
    ) / 3
    return df.assign(
        breakout_growth_potential=breakout_score > breakout_score.quantile(0.9)
    )


def calculate_customer_value_realization_rate(df):
    expected_value = df["customer_ltv"]
    realized_value = df["revenue"] / df["customer_count"]
    time_factor = 1 - np.exp(-df["years_since_founding"] / 5)
    return df.assign(
        value_realization_rate=(realized_value / expected_value) * time_factor
    )


def estimate_team_productivity_index(df):
    revenue_per_employee = df["revenue"] / df["employee_count"]
    growth_factor = 1 + df["growth_rate"]
    funding_efficiency = df["revenue"] / (df["funding_amount"] + 1)
    return df.assign(
        team_productivity_index=(
            revenue_per_employee * growth_factor * funding_efficiency
        )
        ** (1 / 3)
    )


def calculate_market_timing_score(df):
    industry_growth = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.1,
            "Retail": 1.0,
            "Education": 1.2,
        }
    )
    product_readiness = df["product_stage"].map(
        {"Idea": 0.5, "MVP": 0.7, "Beta": 0.9, "Live": 1, "Growth": 1.2, "Mature": 1.1}
    )
    market_adoption = df["product_adoption_rate"]
    return df.assign(
        market_timing_score=industry_growth * product_readiness * market_adoption
    )


def calculate_product_differentiation_score(df):
    feature_uniqueness = df["has_patents"].map({True: 1.2, False: 1})
    market_positioning = 1 + (df["pricing_power_index"] - 1) * 0.5
    customer_perception = df["estimated_customer_sentiment"]
    return df.assign(
        product_differentiation=feature_uniqueness
        * market_positioning
        * customer_perception
    )


def estimate_sales_cycle_length(df):
    base_cycle = 3  # months
    product_complexity_factor = df["product_complexity_score"] * 0.5
    target_market_factor = df["target_market"].map(
        {"Small Business": 0.8, "Enterprise": 1.5, "Consumer": 0.5, "Government": 2}
    )
    price_factor = np.log1p(df["revenue"] / df["customer_count"]) / 5
    return df.assign(
        estimated_sales_cycle=base_cycle
        * (1 + product_complexity_factor)
        * target_market_factor
        * price_factor
    )


def calculate_founder_vision_alignment_score(df):
    market_alignment = df["market_opportunity_score"]
    execution_alignment = df["operational_efficiency"]
    innovation_alignment = df["innovation_index"]
    return df.assign(
        founder_vision_alignment=(
            market_alignment + execution_alignment + innovation_alignment
        )
        / 3
    )


def calculate_product_ecosystem_strength(df):
    partner_network = df["has_partnerships"].map({True: 1.5, False: 1})
    api_extensibility = df["product_complexity_score"] * 0.5
    community_engagement = df["community_engagement_level"]
    return df.assign(
        ecosystem_strength=partner_network * api_extensibility * community_engagement
    )


def estimate_customer_implementation_complexity(df):
    product_complexity = df["product_complexity_score"]
    integration_factor = df["business_model"].map(
        {"B2B": 1.5, "B2C": 0.8, "Marketplace": 1.2, "SaaS": 1, "Hardware": 1.3}
    )
    customization_need = 1 + (1 - df["product_market_fit_score"]) * 0.5
    return df.assign(
        implementation_complexity=product_complexity
        * integration_factor
        * customization_need
    )


def calculate_market_education_burden(df):
    product_novelty = 1 + (1 - df["estimated_market_share"]) * 0.5
    target_market_sophistication = df["target_market"].map(
        {"Small Business": 1.2, "Enterprise": 0.8, "Consumer": 1.5, "Government": 1}
    )
    industry_complexity = df["industry"].map(
        {
            "Technology": 0.9,
            "Healthcare": 1.3,
            "Finance": 1.1,
            "Retail": 0.8,
            "Education": 1,
        }
    )
    return df.assign(
        market_education_burden=product_novelty
        * target_market_sophistication
        * industry_complexity
    )


def identify_potential_category_leaders(df):
    market_share_leadership = df["estimated_market_share"].rank(pct=True)
    innovation_leadership = df["innovation_index"].rank(pct=True)
    brand_strength = df["estimated_brand_value"].rank(pct=True)
    category_leadership_score = (
        market_share_leadership + innovation_leadership + brand_strength
    ) / 3
    return df.assign(
        category_leader_potential=category_leadership_score
        > category_leadership_score.quantile(0.9)
    )


def calculate_startup_adaptability_index(df):
    team_size_factor = 1 / np.log1p(df["employee_count"])
    funding_runway = df["runway_months"] / 12
    product_evolution_speed = df["product_evolution_rate"]
    market_responsiveness = 1 / (1 + df["estimated_sales_cycle"])
    return df.assign(
        adaptability_index=team_size_factor
        * funding_runway
        * product_evolution_speed
        * market_responsiveness
    )


def estimate_customer_success_investment(df):
    base_investment = df["revenue"] * 0.1
    churn_factor = 1 + df["churn_rate"]
    complexity_factor = df["implementation_complexity"]
    return df.assign(
        customer_success_investment=base_investment * churn_factor * complexity_factor
    )


def calculate_product_iteration_frequency(df):
    base_frequency = 12  # iterations per year
    team_size_factor = np.log1p(df["employee_count"]) / 5
    funding_factor = np.log1p(df["funding_amount"]) / 10
    market_pressure = 1 + df["growth_rate"]
    return df.assign(
        iteration_frequency=base_frequency
        * team_size_factor
        * funding_factor
        * market_pressure
    )


def identify_potential_acquihire_targets(df):
    talent_quality = df["talent_density"]
    team_size = df["employee_count"]
    innovation_capability = df["innovation_index"]
    financial_distress = (df["burn_rate"] / df["funding_amount"]).clip(upper=1)
    acquihire_score = (talent_quality * np.log1p(team_size) * innovation_capability) * (
        1 - financial_distress
    )
    return df.assign(
        acquihire_potential=acquihire_score > acquihire_score.quantile(0.9)
    )


def calculate_go_to_market_efficiency(df):
    customer_acquisition_cost = (
        df["burn_rate"] * 0.4 / (df["customer_count"] * df["growth_rate"])
    )
    sales_cycle_length = df["estimated_sales_cycle"]
    conversion_rate = df["growth_rate"] / (df["burn_rate"] * 0.4 / df["revenue"])
    return df.assign(
        gtm_efficiency=1
        / (customer_acquisition_cost * sales_cycle_length / conversion_rate)
    )


def estimate_product_technical_debt(df):
    base_debt = (
        df["years_since_founding"] * 10000
    )  # Assuming $10k of tech debt per year as a base
    growth_factor = 1 + df["growth_rate"] * 2
    refactor_investment = (
        df["burn_rate"] * 0.1
    )  # Assuming 10% of burn rate goes to refactoring
    return df.assign(
        estimated_tech_debt=(base_debt * growth_factor) - refactor_investment
    )


def calculate_startup_moat_strength(df):
    ip_protection = df["has_patents"].map({True: 1.5, False: 1})
    network_effects = df["network_effects_score"]
    switching_costs = 1 / (1 - df["churn_rate"]).clip(upper=10)
    brand_value = df["estimated_brand_value"].rank(pct=True)
    return df.assign(
        moat_strength=(ip_protection * network_effects * switching_costs * brand_value)
        ** 0.25
    )


def identify_potential_second_mover_advantages(df):
    market_maturity = 1 - df["estimated_market_share"]
    learning_from_competitors = 1 - df["product_market_fit_score"]
    resource_availability = df["funding_amount"].rank(pct=True)
    second_mover_score = (
        market_maturity + learning_from_competitors + resource_availability
    ) / 3
    return df.assign(
        second_mover_potential=second_mover_score > second_mover_score.quantile(0.7)
    )


def calculate_customer_upsell_potential(df):
    current_revenue_per_customer = df["revenue"] / df["customer_count"]
    product_breadth = df["product_complexity_score"]
    customer_satisfaction = df["estimated_customer_sentiment"]
    return df.assign(
        upsell_potential=current_revenue_per_customer
        * product_breadth
        * customer_satisfaction
    )


def estimate_startup_culture_strength(df):
    growth_culture = 1 + df["growth_rate"]
    innovation_culture = df["innovation_index"]
    retention_culture = 1 - df["estimated_talent_turnover"]
    return df.assign(
        culture_strength=(growth_culture * innovation_culture * retention_culture)
        ** (1 / 3)
    )


def calculate_product_localization_need(df):
    international_presence = df["location"].map(
        {"US": 0.5, "Europe": 0.7, "Asia": 0.8, "Africa": 0.6, "South America": 0.6}
    )
    product_complexity = df["product_complexity_score"]
    target_market_diversity = df["target_market"].map(
        {"Small Business": 0.7, "Enterprise": 0.9, "Consumer": 1, "Government": 0.5}
    )
    return df.assign(
        localization_need=international_presence
        * product_complexity
        * target_market_diversity
    )


def identify_potential_platform_plays(df):
    ecosystem_strength = df["ecosystem_strength"]
    api_extensibility = df["product_complexity_score"] * 0.5
    market_share = df["estimated_market_share"]
    platform_potential = (ecosystem_strength + api_extensibility + market_share) / 3
    return df.assign(
        platform_play_potential=platform_potential > platform_potential.quantile(0.8)
    )


def calculate_startup_decision_agility(df):
    team_size_factor = 1 / np.log1p(df["employee_count"])
    funding_stage_factor = df["funding_round"].map(
        {"Seed": 1, "Series A": 0.9, "Series B": 0.8, "Series C": 0.7, "Series D+": 0.6}
    )
    product_iteration_speed = df["iteration_frequency"]
    return df.assign(
        decision_agility=team_size_factor
        * funding_stage_factor
        * product_iteration_speed
    )


def estimate_customer_advocacy_strength(df):
    customer_satisfaction = df["estimated_customer_sentiment"]
    product_value = df["value_realization_rate"]
    brand_strength = df["estimated_brand_value"].rank(pct=True)
    return df.assign(
        advocacy_strength=(customer_satisfaction * product_value * brand_strength)
        ** (1 / 3)
    )


def calculate_startup_cash_efficiency(df):
    revenue_to_funding_ratio = df["revenue"] / (df["funding_amount"] + 1)
    burn_multiple = df["burn_multiple"]
    runway_utilization = df["runway_months"] / (df["years_since_founding"] * 12)
    return df.assign(
        cash_efficiency=(revenue_to_funding_ratio / burn_multiple) * runway_utilization
    )


def identify_potential_blitzscalers(df):
    growth_rate = df["growth_rate"]
    market_opportunity = df["market_opportunity_score"]
    network_effects = df["network_effects_score"]
    funding_availability = df["funding_amount"].rank(pct=True)
    blitzscale_score = (
        growth_rate * 2 + market_opportunity + network_effects + funding_availability
    ) / 5
    return df.assign(
        blitzscale_potential=blitzscale_score > blitzscale_score.quantile(0.95)
    )


def calculate_product_stickiness_index(df):
    usage_frequency = df["monthly_active_users"] / df["customer_count"]
    switching_cost = 1 / (1 - df["churn_rate"]).clip(upper=10)
    user_dependency = df["product_complexity_score"] * 0.5
    return df.assign(
        stickiness_index=(usage_frequency * switching_cost * user_dependency) ** (1 / 3)
    )


def estimate_startup_media_presence(df):
    funding_publicity = np.log1p(df["funding_amount"]) / 10
    growth_newsworthiness = df["growth_rate"].clip(lower=0)
    innovation_interest = df["innovation_index"]
    market_impact = df["ecosystem_impact_score"]
    return df.assign(
        media_presence=(
            funding_publicity
            + growth_newsworthiness
            + innovation_interest
            + market_impact
        )
        / 4
    )


def calculate_founder_investor_alignment(df):
    equity_retention = 1 - df["funding_round"].map(
        {
            "Seed": 0.1,
            "Series A": 0.2,
            "Series B": 0.3,
            "Series C": 0.4,
            "Series D+": 0.5,
        }
    )
    board_composition = df["funding_round"].map(
        {
            "Seed": 0.9,
            "Series A": 0.8,
            "Series B": 0.7,
            "Series C": 0.6,
            "Series D+": 0.5,
        }
    )
    strategic_alignment = df["founder_vision_alignment"]
    return df.assign(
        founder_investor_alignment=(
            equity_retention * board_composition * strategic_alignment
        )
        ** (1 / 3)
    )


def identify_potential_industry_bridges(df):
    cross_industry_applicability = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.2,
            "Finance": 1.3,
            "Retail": 1.1,
            "Education": 1.4,
        }
    )
    product_adaptability = 1 / df["product_complexity_score"]
    market_diversity = df["customer_segment_diversity"]
    bridge_score = (
        cross_industry_applicability * product_adaptability * market_diversity
    )
    return df.assign(
        industry_bridge_potential=bridge_score > bridge_score.quantile(0.9)
    )


def calculate_startup_talent_magnetism(df):
    growth_appeal = df["growth_rate"].clip(lower=0)
    innovation_appeal = df["innovation_index"]
    compensation_potential = (df["funding_amount"] / df["employee_count"]).rank(
        pct=True
    )
    work_culture = df["culture_strength"]
    return df.assign(
        talent_magnetism=(
            growth_appeal + innovation_appeal + compensation_potential + work_culture
        )
        / 4
    )


def estimate_product_market_saturation(df):
    market_penetration = df["estimated_market_share"]
    growth_slowdown = 1 - df["growth_rate"].clip(lower=0, upper=1)
    competition_intensity = 1 / (df["pricing_power_index"] + 0.1)
    return df.assign(
        market_saturation=(market_penetration + growth_slowdown + competition_intensity)
        / 3
    )


def calculate_startup_pivot_readiness(df):
    financial_flexibility = df["runway_months"] / 12
    team_adaptability = 1 / (df["years_since_founding"] + 1)
    market_pressure = 1 - df["product_market_fit_score"]
    return df.assign(
        pivot_readiness=financial_flexibility * team_adaptability * market_pressure
    )


def identify_potential_moonshots(df):
    market_size_potential = np.log1p(df["estimated_tam"])
    technological_ambition = df["innovation_index"]
    long_term_vision = df["runway_months"] / 12
    risk_tolerance = df["burn_rate"] / df["funding_amount"]
    moonshot_score = (
        market_size_potential
        + technological_ambition
        + long_term_vision
        + risk_tolerance
    ) / 4
    return df.assign(moonshot_potential=moonshot_score > moonshot_score.quantile(0.98))


def calculate_startup_ecosystem_contribution(df):
    job_creation = df["employee_count"] / 100
    innovation_spillover = df["innovation_index"]
    economic_impact = df["revenue"] / 1e6
    return df.assign(
        ecosystem_contribution=(job_creation + innovation_spillover + economic_impact)
        / 3
    )


def estimate_product_feature_bloat_risk(df):
    feature_complexity = df["product_complexity_score"]
    development_speed = df["product_evolution_rate"]
    market_feedback_lag = df["estimated_sales_cycle"]
    return df.assign(
        feature_bloat_risk=feature_complexity * development_speed * market_feedback_lag
    )


def calculate_startup_global_scalability(df):
    product_localization = 1 / df["localization_need"]
    market_expansion_potential = df["market_expansion_potential"]
    operational_readiness = df["operational_efficiency"]
    funding_adequacy = np.log1p(df["funding_amount"]) / 10
    return df.assign(
        global_scalability=(
            product_localization
            * market_expansion_potential
            * operational_readiness
            * funding_adequacy
        )
        ** 0.25
    )


def identify_potential_value_chain_disruptors(df):
    vertical_integration = df["business_model"].map(
        {"B2B": 0.7, "B2C": 0.8, "Marketplace": 1, "SaaS": 0.9, "Hardware": 0.6}
    )
    cost_efficiency = df["operational_efficiency"]
    innovation_impact = df["innovation_index"]
    market_reception = df["product_market_fit_score"]
    disruptor_score = (
        vertical_integration * cost_efficiency * innovation_impact * market_reception
    ) ** 0.25
    return df.assign(
        value_chain_disruptor_potential=disruptor_score > disruptor_score.quantile(0.9)
    )


def calculate_startup_regulatory_navigation_ability(df):
    regulatory_burden = df["regulatory_burden"]
    legal_resources = (df["funding_amount"] * 0.05).clip(
        upper=1e6
    )  # Assuming 5% of funding goes to legal, capped at $1M
    industry_experience = df["years_since_founding"]
    compliance_score = (
        1
        / (regulatory_burden + 0.1)
        * np.log1p(legal_resources)
        * np.log1p(industry_experience)
    )
    return df.assign(regulatory_navigation_ability=compliance_score)


def estimate_customer_data_leverage(df):
    data_volume = df["monthly_active_users"] * df["years_since_founding"]
    data_variety = df["product_complexity_score"]
    data_utilization = df["innovation_index"]
    return df.assign(
        data_leverage=(np.log1p(data_volume) * data_variety * data_utilization)
        ** (1 / 3)
    )


def calculate_startup_talent_retention_power(df):
    compensation_competitiveness = (df["funding_amount"] / df["employee_count"]).rank(
        pct=True
    )
    growth_opportunities = df["growth_rate"].clip(lower=0)
    work_life_balance = 1 / df["burn_multiple"].clip(lower=0.1)
    mission_alignment = df["founder_vision_alignment"]
    return df.assign(
        talent_retention_power=(
            compensation_competitiveness
            + growth_opportunities
            + work_life_balance
            + mission_alignment
        )
        / 4
    )


def identify_potential_market_educators(df):
    product_novelty = 1 - df["estimated_market_share"]
    thought_leadership = df["innovation_index"]
    content_production = df["marketing_spend"] / df["burn_rate"]
    educator_score = (product_novelty + thought_leadership + content_production) / 3
    return df.assign(
        market_educator_potential=educator_score > educator_score.quantile(0.8)
    )


def calculate_startup_partnership_leverage(df):
    market_position = df["estimated_market_share"].rank(pct=True)
    product_complementarity = 1 - df["product_market_fit_score"]
    resource_availability = df["funding_amount"].rank(pct=True)
    strategic_alignment = df["founder_vision_alignment"]
    return df.assign(
        partnership_leverage=(
            market_position
            + product_complementarity
            + resource_availability
            + strategic_alignment
        )
        / 4
    )


def estimate_product_technical_moat(df):
    ip_strength = df["has_patents"].map({True: 1.5, False: 1})
    engineering_talent = df["talent_density"]
    tech_stack_complexity = df["tech_stack_complexity"]
    data_advantage = df["data_leverage"]
    return df.assign(
        technical_moat=(
            ip_strength * engineering_talent * tech_stack_complexity * data_advantage
        )
        ** 0.25
    )


def calculate_startup_innovation_velocity(df):
    rd_intensity = df["burn_rate"] * 0.3 / df["revenue"]
    talent_quality = df["talent_density"]
    product_iteration_speed = df["iteration_frequency"]
    return df.assign(
        innovation_velocity=rd_intensity * talent_quality * product_iteration_speed
    )


def estimate_startup_brand_equity(df):
    market_presence = df["estimated_market_share"]
    customer_loyalty = 1 - df["churn_rate"]
    media_coverage = df["media_presence"]
    return df.assign(
        brand_equity=(market_presence * customer_loyalty * media_coverage) ** (1 / 3)
    )


def calculate_startup_operational_resilience(df):
    financial_buffer = df["runway_months"] / 12
    team_adaptability = 1 / np.log1p(df["employee_count"])
    business_model_diversity = df["customer_segment_diversity"]
    return df.assign(
        operational_resilience=financial_buffer
        * team_adaptability
        * business_model_diversity
    )


def estimate_startup_network_effect_strength(df):
    user_base_factor = np.log1p(df["monthly_active_users"])
    engagement_rate = df["monthly_active_users"] / df["customer_count"]
    virality_factor = df["estimated_viral_coefficient"]
    return df.assign(
        network_effect_strength=user_base_factor * engagement_rate * virality_factor
    )


def calculate_startup_customer_acquisition_efficiency(df):
    cac = df["burn_rate"] * 0.4 / (df["customer_count"] * df["growth_rate"])
    ltv = df["customer_ltv"]
    payback_period = cac / (ltv / 24)  # Assuming 24-month customer lifetime
    return df.assign(acquisition_efficiency=1 / (cac * payback_period))


def identify_potential_industry_transformers(df):
    market_impact = df["ecosystem_impact_score"]
    innovation_level = df["innovation_index"]
    scalability = df["global_scalability"]
    transformer_score = (market_impact + innovation_level + scalability) / 3
    return df.assign(
        industry_transformer_potential=transformer_score
        > transformer_score.quantile(0.9)
    )


def estimate_startup_pricing_power(df):
    market_share = df["estimated_market_share"]
    product_uniqueness = df["product_differentiation"]
    customer_value_perception = df["value_realization_rate"]
    return df.assign(
        pricing_power=(market_share * product_uniqueness * customer_value_perception)
        ** (1 / 3)
    )


def calculate_startup_talent_development_capacity(df):
    learning_budget = (
        df["burn_rate"] * 0.05
    )  # Assuming 5% of burn rate goes to learning and development
    growth_opportunities = df["growth_rate"].clip(lower=0)
    mentorship_potential = df["founder_experience"].map(
        {"First-time": 0.7, "Serial": 1, "Industry Expert": 1.2}
    )
    return df.assign(
        talent_development_capacity=(
            np.log1p(learning_budget) * growth_opportunities * mentorship_potential
        )
        ** (1 / 3)
    )


def identify_potential_market_timing_masters(df):
    market_growth = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.1,
            "Retail": 1.0,
            "Education": 1.2,
        }
    )
    product_readiness = df["product_market_fit_score"]
    execution_speed = df["iteration_frequency"]
    timing_score = (market_growth * product_readiness * execution_speed) ** (1 / 3)
    return df.assign(market_timing_mastery=timing_score > timing_score.quantile(0.9))


def estimate_startup_customer_success_capacity(df):
    support_staff_ratio = 0.1
    estimated_support_staff = df["employee_count"] * support_staff_ratio
    customer_complexity = df["implementation_complexity"]
    product_quality = df["product_market_fit_score"]
    return df.assign(
        customer_success_capacity=(estimated_support_staff / customer_complexity)
        * product_quality
    )


def calculate_startup_cash_conversion_cycle(df):
    days_sales_outstanding = 30  # Assuming 30 days on average
    days_payable_outstanding = 45  # Assuming 45 days on average
    inventory_days = df["business_model"].map(
        {"B2B": 0, "B2C": 15, "Marketplace": 0, "SaaS": 0, "Hardware": 30}
    )
    return df.assign(
        cash_conversion_cycle=days_sales_outstanding
        + inventory_days
        - days_payable_outstanding
    )


def identify_potential_blue_ocean_strategists(df):
    market_creation = 1 - df["estimated_market_share"]
    value_innovation = df["product_differentiation"]
    cost_leadership = df["operational_efficiency"]
    blue_ocean_score = (market_creation + value_innovation + cost_leadership) / 3
    return df.assign(
        blue_ocean_potential=blue_ocean_score > blue_ocean_score.quantile(0.9)
    )


def estimate_startup_intellectual_property_value(df):
    patent_value = df["has_patents"].map(
        {True: 1e6, False: 0}
    )  # Assuming $1M value for having patents
    trade_secret_value = (
        df["burn_rate"] * df["years_since_founding"] * 0.1
    )  # Assuming 10% of total burn contributes to trade secrets
    brand_value = df["estimated_brand_value"]
    return df.assign(ip_value=patent_value + trade_secret_value + brand_value)


def calculate_startup_customer_engagement_depth(df):
    usage_frequency = df["monthly_active_users"] / df["customer_count"]
    feature_adoption = (
        df["product_complexity_score"] * 0.5
    )  # Assuming 50% feature adoption on average
    feedback_rate = 0.1  # Assuming 10% of customers provide regular feedback
    return df.assign(
        engagement_depth=usage_frequency * feature_adoption * feedback_rate
    )


def identify_potential_talent_incubators(df):
    alumni_success = df["talent_density"]
    learning_environment = df["talent_development_capacity"]
    industry_reputation = df["ecosystem_impact_score"]
    incubator_score = (alumni_success + learning_environment + industry_reputation) / 3
    return df.assign(
        talent_incubator_potential=incubator_score > incubator_score.quantile(0.8)
    )


def estimate_startup_decision_making_agility(df):
    org_flatness = 1 / np.log1p(df["employee_count"])
    data_driven_culture = df["data_leverage"]
    experimentation_rate = df["iteration_frequency"]
    return df.assign(
        decision_agility=org_flatness * data_driven_culture * experimentation_rate
    )


def calculate_startup_ecosystem_influence(df):
    market_leadership = df["estimated_market_share"].rank(pct=True)
    thought_leadership = df["innovation_index"]
    partnership_strength = df["partnership_leverage"]
    return df.assign(
        ecosystem_influence=(
            market_leadership + thought_leadership + partnership_strength
        )
        / 3
    )


def identify_potential_profit_formula_innovators(df):
    revenue_model_uniqueness = 1 - df["estimated_market_share"]
    cost_structure_efficiency = df["operational_efficiency"]
    value_capture_effectiveness = df["pricing_power"]
    innovator_score = (
        revenue_model_uniqueness
        + cost_structure_efficiency
        + value_capture_effectiveness
    ) / 3
    return df.assign(
        profit_formula_innovator_potential=innovator_score
        > innovator_score.quantile(0.9)
    )


def estimate_startup_cultural_alignment_strength(df):
    vision_clarity = df["founder_vision_alignment"]
    value_consistency = 1 - df["estimated_talent_turnover"]
    mission_driven_factor = df["social_impact_score"]
    return df.assign(
        cultural_alignment=(vision_clarity + value_consistency + mission_driven_factor)
        / 3
    )


def calculate_startup_market_creation_ability(df):
    product_novelty = 1 - df["product_market_fit_score"]
    customer_education_investment = (
        df["burn_rate"] * 0.15
    )  # Assuming 15% of burn rate goes to market education
    first_mover_advantage = 1 / (df["years_since_founding"] + 1)
    return df.assign(
        market_creation_ability=product_novelty
        * np.log1p(customer_education_investment)
        * first_mover_advantage
    )


def identify_potential_efficiency_disruptors(df):
    cost_advantage = 1 / df["burn_multiple"]
    operational_innovation = df["operational_efficiency"]
    scalability = df["global_scalability"]
    disruptor_score = (cost_advantage + operational_innovation + scalability) / 3
    return df.assign(
        efficiency_disruptor_potential=disruptor_score > disruptor_score.quantile(0.9)
    )


def estimate_startup_pivot_success_probability(df):
    financial_runway = df["runway_months"] / 12
    team_adaptability = 1 / np.log1p(df["years_since_founding"])
    market_opportunity = df["market_opportunity_score"]
    return df.assign(
        pivot_success_probability=financial_runway
        * team_adaptability
        * market_opportunity
    )


def calculate_startup_customer_feedback_loop_strength(df):
    feedback_collection_rate = 0.1  # Assuming 10% of customers provide feedback
    iteration_speed = df["iteration_frequency"]
    customer_centricity = df["customer_success_capacity"]
    return df.assign(
        feedback_loop_strength=feedback_collection_rate
        * iteration_speed
        * customer_centricity
    )


def identify_potential_platform_economy_players(df):
    ecosystem_building = df["partnership_leverage"]
    value_creation_for_others = 1 - df["value_realization_rate"]
    scalability = df["global_scalability"]
    platform_score = (ecosystem_building + value_creation_for_others + scalability) / 3
    return df.assign(
        platform_economy_potential=platform_score > platform_score.quantile(0.9)
    )


def estimate_startup_regulatory_arbitrage_potential(df):
    regulatory_burden = df["regulatory_burden"]
    international_presence = df["location"].map(
        {"US": 0.5, "Europe": 0.7, "Asia": 0.8, "Africa": 0.6, "South America": 0.6}
    )
    legal_innovation = (
        df["innovation_index"] * 0.5
    )  # Assuming 50% of innovation applies to legal domain
    return df.assign(
        regulatory_arbitrage_potential=(1 / regulatory_burden)
        * international_presence
        * legal_innovation
    )


def calculate_startup_talent_attraction_gravity(df):
    compensation_competitiveness = (df["funding_amount"] / df["employee_count"]).rank(
        pct=True
    )
    learning_opportunity = df["talent_development_capacity"]
    startup_prestige = df["ecosystem_impact_score"]
    return df.assign(
        talent_attraction_gravity=(
            compensation_competitiveness + learning_opportunity + startup_prestige
        )
        / 3
    )


def identify_potential_customer_experience_revolutionizers(df):
    product_delight = df["product_market_fit_score"]
    service_quality = df["customer_success_capacity"]
    personalization_level = df["data_leverage"]
    revolutionizer_score = (
        product_delight + service_quality + personalization_level
    ) / 3
    return df.assign(
        cx_revolutionizer_potential=revolutionizer_score
        > revolutionizer_score.quantile(0.9)
    )


def estimate_startup_m_and_a_attractiveness(df):
    strategic_value = df["innovation_index"]
    financial_health = df["runway_months"] / 12
    market_position = df["estimated_market_share"].rank(pct=True)
    synergy_potential = df["ecosystem_impact_score"]
    return df.assign(
        m_and_a_attractiveness=(
            strategic_value + financial_health + market_position + synergy_potential
        )
        / 4
    )


def calculate_startup_product_evolution_velocity(df):
    iteration_frequency = df["iteration_frequency"]
    development_team_size = (
        df["employee_count"] * 0.4
    )  # Assuming 40% of employees are in product development
    customer_feedback_integration = df["feedback_loop_strength"]
    return df.assign(
        product_evolution_velocity=iteration_frequency
        * np.log1p(development_team_size)
        * customer_feedback_integration
    )


def identify_potential_industry_standard_setters(df):
    market_influence = df["ecosystem_influence"]
    innovation_leadership = df["innovation_index"]
    partnership_network = df["partnership_leverage"]
    standard_setter_score = (
        market_influence + innovation_leadership + partnership_network
    ) / 3
    return df.assign(
        industry_standard_setter_potential=standard_setter_score
        > standard_setter_score.quantile(0.95)
    )


def estimate_startup_remote_work_adaptability(df):
    digital_infrastructure = df["tech_stack_complexity"]
    team_distribution = df["location"].map(
        {"US": 0.8, "Europe": 0.9, "Asia": 0.7, "Africa": 0.6, "South America": 0.7}
    )
    collaboration_tools_investment = (
        df["burn_rate"] * 0.05
    )  # Assuming 5% of burn rate goes to collaboration tools
    return df.assign(
        remote_work_adaptability=digital_infrastructure
        * team_distribution
        * np.log1p(collaboration_tools_investment)
    )


def calculate_startup_customer_education_effectiveness(df):
    content_production = (
        df["burn_rate"] * 0.1
    )  # Assuming 10% of burn rate goes to content production
    audience_reach = np.log1p(df["monthly_active_users"])
    product_complexity = df["product_complexity_score"]
    return df.assign(
        customer_education_effectiveness=(np.log1p(content_production) * audience_reach)
        / product_complexity
    )


def estimate_startup_social_impact_potential(df):
    industry_impact = df["industry"].map(
        {
            "Technology": 0.8,
            "Healthcare": 1.2,
            "Finance": 0.7,
            "Retail": 0.6,
            "Education": 1.1,
        }
    )
    sustainability_focus = df["has_sustainability_initiatives"].map(
        {True: 1.5, False: 1}
    )
    scalability = df["global_scalability"]
    return df.assign(
        social_impact_potential=industry_impact * sustainability_focus * scalability
    )


def calculate_startup_knowledge_capital_index(df):
    patent_value = df["has_patents"].map({True: 1.5, False: 1})
    employee_expertise = df["talent_density"]
    rd_intensity = df["burn_rate"] * 0.3 / df["revenue"]
    return df.assign(
        knowledge_capital_index=(patent_value * employee_expertise * rd_intensity)
        ** (1 / 3)
    )


def identify_potential_niche_market_dominators(df):
    market_focus = 1 - df["customer_segment_diversity"]
    market_share = df["estimated_market_share"]
    product_specialization = df["product_complexity_score"]
    dominator_score = (market_focus * market_share * product_specialization) ** (1 / 3)
    return df.assign(
        niche_dominator_potential=dominator_score > dominator_score.quantile(0.9)
    )


def estimate_startup_crisis_resilience(df):
    financial_buffer = df["runway_months"] / 12
    business_model_adaptability = 1 / df["product_complexity_score"]
    customer_diversity = df["customer_segment_diversity"]
    remote_work_capability = df["remote_work_adaptability"]
    return df.assign(
        crisis_resilience=(
            financial_buffer
            * business_model_adaptability
            * customer_diversity
            * remote_work_capability
        )
        ** 0.25
    )


def calculate_startup_innovation_efficiency(df):
    innovation_output = df["innovation_index"]
    rd_investment = df["burn_rate"] * 0.3  # Assuming 30% of burn rate goes to R&D
    time_factor = np.log1p(df["years_since_founding"])
    return df.assign(
        innovation_efficiency=innovation_output
        / (np.log1p(rd_investment) * time_factor)
    )


def estimate_startup_market_timing_accuracy(df):
    product_readiness = df["product_market_fit_score"]
    market_growth = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.1,
            "Retail": 1.0,
            "Education": 1.2,
        }
    )
    execution_speed = df["iteration_frequency"]
    return df.assign(
        market_timing_accuracy=product_readiness * market_growth * execution_speed
    )


def calculate_startup_customer_acquisition_velocity(df):
    marketing_spend = (
        df["burn_rate"] * 0.3
    )  # Assuming 30% of burn rate goes to marketing
    viral_coefficient = df["estimated_viral_coefficient"]
    product_appeal = df["product_market_fit_score"]
    return df.assign(
        customer_acquisition_velocity=(
            np.log1p(marketing_spend) * viral_coefficient * product_appeal
        )
    )


def estimate_startup_ecosystem_contribution_index(df):
    job_creation = df["employee_count"] / 100
    innovation_spillover = df["innovation_index"]
    economic_impact = df["revenue"] / 1e6
    return df.assign(
        ecosystem_contribution_index=(
            job_creation + innovation_spillover + economic_impact
        )
        / 3
    )


def calculate_startup_product_market_fit_velocity(df):
    iteration_speed = df["iteration_frequency"]
    customer_feedback = df["feedback_loop_strength"]
    market_responsiveness = 1 / (1 + df["estimated_sales_cycle"])
    return df.assign(
        product_market_fit_velocity=iteration_speed
        * customer_feedback
        * market_responsiveness
    )


def estimate_startup_brand_equity_growth(df):
    current_brand_value = df["estimated_brand_value"]
    market_penetration_rate = df["growth_rate"].clip(lower=0)
    customer_loyalty = 1 - df["churn_rate"]
    return df.assign(
        brand_equity_growth=current_brand_value
        * market_penetration_rate
        * customer_loyalty
    )


def identify_potential_unicorn_candidates(df):
    growth_trajectory = df["growth_rate"] ** 2
    market_opportunity = df["market_opportunity_score"]
    funding_momentum = np.log1p(df["funding_amount"]) / df["years_since_founding"]
    unicorn_score = (growth_trajectory * market_opportunity * funding_momentum) ** (
        1 / 3
    )
    return df.assign(
        unicorn_candidate_potential=unicorn_score > unicorn_score.quantile(0.98)
    )


def calculate_startup_operational_efficiency_index(df):
    revenue_per_employee = df["revenue"] / df["employee_count"]
    burn_rate_efficiency = df["revenue"] / df["burn_rate"]
    scalability = df["global_scalability"]
    return df.assign(
        operational_efficiency_index=(
            revenue_per_employee * burn_rate_efficiency * scalability
        )
        ** (1 / 3)
    )


def estimate_startup_customer_lifetime_value_growth(df):
    current_ltv = df["customer_ltv"]
    cross_sell_opportunity = df["product_complexity_score"]
    customer_satisfaction = df["estimated_customer_sentiment"]
    return df.assign(
        ltv_growth_potential=current_ltv
        * cross_sell_opportunity
        * customer_satisfaction
    )


def calculate_startup_talent_retention_index(df):
    compensation_competitiveness = (df["funding_amount"] / df["employee_count"]).rank(
        pct=True
    )
    growth_opportunity = df["growth_rate"].clip(lower=0)
    work_culture = df["culture_strength"]
    return df.assign(
        talent_retention_index=(
            compensation_competitiveness + growth_opportunity + work_culture
        )
        / 3
    )


def estimate_startup_pricing_power_potential(df):
    market_share = df["estimated_market_share"]
    product_uniqueness = df["product_differentiation"]
    customer_value_perception = df["value_realization_rate"]
    return df.assign(
        pricing_power_potential=(
            market_share * product_uniqueness * customer_value_perception
        )
        ** (1 / 3)
    )


def calculate_startup_customer_success_index(df):
    onboarding_efficiency = 1 / df["implementation_complexity"]
    support_quality = df["customer_success_capacity"]
    product_ease_of_use = 1 / df["product_complexity_score"]
    return df.assign(
        customer_success_index=(
            onboarding_efficiency * support_quality * product_ease_of_use
        )
        ** (1 / 3)
    )


def estimate_startup_market_penetration_velocity(df):
    growth_rate = df["growth_rate"].clip(lower=0)
    market_size = df["estimated_tam"]
    go_to_market_efficiency = df["gtm_efficiency"]
    return df.assign(
        market_penetration_velocity=(
            growth_rate * np.log1p(market_size) * go_to_market_efficiency
        )
        ** (1 / 3)
    )


def identify_potential_industry_consolidators(df):
    market_share = df["estimated_market_share"]
    financial_strength = df["cash_reserves"].rank(pct=True)
    operational_efficiency = df["operational_efficiency"]
    consolidator_score = (
        market_share + financial_strength + operational_efficiency
    ) / 3
    return df.assign(
        industry_consolidator_potential=consolidator_score
        > consolidator_score.quantile(0.9)
    )


def calculate_startup_innovation_to_market_index(df):
    innovation_rate = df["innovation_index"]
    time_to_market = 1 / df["iteration_frequency"]
    market_reception = df["product_market_fit_score"]
    return df.assign(
        innovation_to_market_index=(innovation_rate / time_to_market) * market_reception
    )


def estimate_startup_customer_acquisition_cost_efficiency(df):
    marketing_spend = (
        df["burn_rate"] * 0.3
    )  # Assuming 30% of burn rate goes to marketing
    new_customers = df["customer_count"] * df["growth_rate"]
    ltv = df["customer_ltv"]
    return df.assign(cac_efficiency=ltv / (marketing_spend / new_customers))


def calculate_startup_product_development_efficiency(df):
    feature_delivery_rate = df["iteration_frequency"]
    development_team_size = (
        df["employee_count"] * 0.4
    )  # Assuming 40% of employees are in product development
    product_complexity = df["product_complexity_score"]
    return df.assign(
        product_development_efficiency=(
            feature_delivery_rate * np.log1p(development_team_size)
        )
        / product_complexity
    )


def estimate_startup_market_timing_advantage(df):
    product_readiness = df["product_market_fit_score"]
    market_growth = df["industry"].map(
        {
            "Technology": 1.5,
            "Healthcare": 1.3,
            "Finance": 1.1,
            "Retail": 1.0,
            "Education": 1.2,
        }
    )
    execution_speed = df["iteration_frequency"]
    return df.assign(
        market_timing_advantage=product_readiness * market_growth * execution_speed
    )


def calculate_startup_customer_retention_strength(df):
    product_stickiness = df["product_stickiness"]
    customer_success_quality = df["customer_success_index"]
    switching_cost = 1 / (1 - df["churn_rate"]).clip(upper=10)
    return df.assign(
        customer_retention_strength=(
            product_stickiness * customer_success_quality * switching_cost
        )
        ** (1 / 3)
    )


def estimate_startup_viral_growth_potential(df):
    viral_coefficient = df["estimated_viral_coefficient"]
    user_engagement = df["monthly_active_users"] / df["customer_count"]
    product_share_ability = (
        df["product_complexity_score"] * 0.5
    )  # Assuming simpler products are more shareable
    return df.assign(
        viral_growth_potential=viral_coefficient
        * user_engagement
        / product_share_ability
    )


def calculate_startup_innovation_roi(df):
    innovation_output = df["innovation_index"]
    rd_investment = df["burn_rate"] * 0.3  # Assuming 30% of burn rate goes to R&D
    time_to_market = 1 / df["iteration_frequency"]
    return df.assign(
        innovation_roi=innovation_output / (np.log1p(rd_investment) * time_to_market)
    )


def estimate_startup_market_education_burden(df):
    product_novelty = 1 - df["product_market_fit_score"]
    market_sophistication = df["industry"].map(
        {
            "Technology": 0.8,
            "Healthcare": 1.2,
            "Finance": 1.0,
            "Retail": 0.7,
            "Education": 0.9,
        }
    )
    customer_learning_curve = df["product_complexity_score"]
    return df.assign(
        market_education_burden=product_novelty
        * market_sophistication
        * customer_learning_curve
    )


def calculate_startup_customer_feedback_utilization(df):
    feedback_collection_rate = df["feedback_loop_strength"]
    product_iteration_speed = df["iteration_frequency"]
    customer_centricity = df["customer_success_capacity"]
    return df.assign(
        customer_feedback_utilization=feedback_collection_rate
        * product_iteration_speed
        * customer_centricity
    )


def estimate_startup_pivot_readiness(df):
    financial_runway = df["runway_months"] / 12
    team_adaptability = 1 / np.log1p(df["years_since_founding"])
    market_exploration = 1 - df["product_market_fit_score"]
    return df.assign(
        pivot_readiness=financial_runway * team_adaptability * market_exploration
    )


def identify_potential_category_kings(df):
    market_share = df["estimated_market_share"]
    brand_strength = df["estimated_brand_value"].rank(pct=True)
    innovation_leadership = df["innovation_index"]
    category_king_score = (market_share + brand_strength + innovation_leadership) / 3
    return df.assign(
        category_king_potential=category_king_score > category_king_score.quantile(0.98)
    )


def calculate_startup_talent_leverage(df):
    revenue_per_employee = df["revenue"] / df["employee_count"]
    employee_productivity = df["operational_efficiency"]
    talent_quality = df["talent_density"]
    return df.assign(
        talent_leverage=(revenue_per_employee * employee_productivity * talent_quality)
        ** (1 / 3)
    )


def estimate_startup_product_ecosystem_strength(df):
    api_extensibility = df["product_complexity_score"] * 0.5
    partner_network = df["partnership_leverage"]
    developer_adoption = df["ecosystem_impact_score"]
    return df.assign(
        product_ecosystem_strength=(
            api_extensibility * partner_network * developer_adoption
        )
        ** (1 / 3)
    )


def estimate_startup_market_education_effectiveness(df):
    content_production = df["customer_education_effectiveness"]
    audience_reach = np.log1p(df["monthly_active_users"])
    product_complexity = df["product_complexity_score"]
    return df.assign(
        market_education_effectiveness=(content_production * audience_reach)
        / product_complexity
    )


def calculate_startup_customer_success_leverage(df):
    customer_satisfaction = df["estimated_customer_sentiment"]
    product_stickiness = df["product_stickiness"]
    support_efficiency = df["customer_success_capacity"]
    return df.assign(
        customer_success_leverage=(
            customer_satisfaction * product_stickiness * support_efficiency
        )
        ** (1 / 3)
    )


def estimate_startup_regulatory_navigation_capability(df):
    regulatory_burden = df["regulatory_burden"]
    legal_expertise = (df["funding_amount"] * 0.05).clip(
        upper=1e6
    )  # Assuming 5% of funding goes to legal, capped at $1M
    compliance_track_record = 1 - df["regulatory_risk_score"]
    return df.assign(
        regulatory_navigation_capability=(1 / regulatory_burden)
        * np.log1p(legal_expertise)
        * compliance_track_record
    )


def identify_potential_moonshot_innovators(df):
    vision_magnitude = df["founder_vision_alignment"]
    risk_tolerance = df["burn_rate"] / df["funding_amount"]
    innovation_capacity = df["innovation_index"]
    moonshot_score = (vision_magnitude * risk_tolerance * innovation_capacity) ** (
        1 / 3
    )
    return df.assign(
        moonshot_innovator_potential=moonshot_score > moonshot_score.quantile(0.98)
    )
