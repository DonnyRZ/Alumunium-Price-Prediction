# Overview of Naive Forecast
That paper talks about this a lot. It is not the only topic of the paper, but it is one of its central warnings: in time-series forecasting, a sophisticated ML model can end up behaving almost the same as a naïve forecast, and unless you evaluate correctly, you can fool yourself into thinking the complex model is adding value when it is not. The paper makes this point most directly in Sections 3.2 and 3.3, especially around the examples on pp. 8–12 and again in the concluding guidelines. 

The first important thing is that there is not one universally fixed name for this phenomenon. People usually describe it as one of these:

* failing to beat the naïve or persistence benchmark,
* random-walk or unit-root behavior,
* a model collapsing to persistence,
* no-change forecast dominance.

In the paper, the framing is mainly: **for some time series, especially those with unit-root / random-walk-like behavior, the naïve forecast is already theoretically optimal or extremely hard to beat**, so complex models often end up producing forecasts that look almost the same. 

What the naïve forecast is:
the paper defines it as the forecast that simply uses the last observed value as the future prediction. In forecasting this is also called the persistence model or no-change model. For a random walk, the data-generating process is ( y_{t+1}=y_t+\varepsilon_t ), and the corresponding optimal point forecast is ( \hat y_{t+h}=y_t ). The paper explicitly says that if the series has no predictable structure beyond that unit-root behavior, then more complex forecasting methods have no true predictive power beyond the naïve forecast, and any measured superiority is just chance and should disappear on sufficiently large datasets. That is the heart of the issue you are asking about. It is stated very directly around pp. 9–10. 

So the key idea is this:

A complex model matching the naïve forecast is **not always a failure**. Sometimes it is exactly what a good model should do, because the series does not contain exploitable predictive structure beyond “tomorrow looks like today.” The paper even notes that many practical series show strongly integrated behavior close to random walks, such as stock market data, wind power, and wind speed. In those cases, the naïve forecast is a serious benchmark, not a toy baseline. 

Why complex ML models end up looking naïve

One reason is the underlying process itself. If the series is close to a random walk, there may simply be nothing to learn beyond the last value. Another reason is model design: many autoregressive ML models are trained to predict the next value from recent lags, and if lag-1 dominates while the remaining signal is weak, the safest loss-minimizing behavior is to copy the most recent observation. That makes the model look smart, but functionally it is just persistence.

The paper also shows a second version of this problem. On p. 8 it gives an example where ML models are trained on a unit-root non-stationary series **without** appropriate preprocessing, and then they struggle to predict values outside the range seen in training. Then on pp. 9–10 it shows that even **with** differencing as preprocessing, several models still end up behaving very similarly to the naïve forecast, because the differenced series contains little extra signal. In other words: preprocessing can stop the model from failing badly, but it does not magically create predictability that was never there. 

This leads to one of the most important distinctions:

There is a **benign** version of the phenomenon and a **bad** version.

The benign version is: the data are near-random-walk, and the complex model correctly discovers that the best forecast is basically persistence.

The bad version is: the model looks like naïve because of weak modeling choices, poor preprocessing, bad scaling, too little data, misaligned loss/metric, or because the evaluation setup is misleading.

The paper is mainly concerned with telling those two apart. 

Why this fools people

Forecast plots are especially deceptive here. The paper warns that on rolling-origin evaluation, the naïve forecast gets updated each time with the latest observation, so it visually “tracks” the actual series and can look very impressive. Our eyes focus on how the forecast line follows the path of the data, but for error evaluation the relevant thing is the vertical distance, not the horizontal closeness. The paper says forecast plots should mostly be for sanity checking; decisions should be based on proper error measures and benchmarks, not on visual appeal. This is discussed on pp. 11–12 with Figure 7. 

That point matters a lot: if a deep model and the naïve forecast produce very similar plots, the natural temptation is to say the deep model “understands the dynamics.” But the paper’s point is that on integrated series, many methods will visually follow the series in almost the same way, so the plot by itself tells you very little. 

What the paper says you should do instead

You must benchmark against the simplest valid baseline. For non-seasonal integrated series, that is the naïve forecast. For seasonal series, it should usually be the seasonal naïve. The paper says explicitly that without comparing against the naïve benchmark, the quality of a more complex model cannot be meaningfully assessed. It also recommends relative or scaled error measures that compare performance against such simple baselines. 

This is really the broader lesson of the paper: a complex model is not impressive because it makes smooth-looking forecasts. It is impressive only if it shows reliable skill over the right benchmark under the right evaluation setup. 

Why apparent gains over naïve can still be fake

The paper gives several reasons:

First, pure luck. On a random-walk-like series, a complex model may beat naïve on a small test sample, but that can be a spurious result.

Second, wrong metrics. Some error measures can distort what “better” means, especially under trend, seasonality, intermittency, or level shifts. The paper spends a lot of effort showing that there is no universally safe forecasting metric; the chosen measure must fit the properties of the series. 

Third, data leakage. In forecasting, leakage can happen easily through rolling-origin evaluation, normalization, smoothing, decomposition, or feature extraction done using the whole series before splitting. Leakage can make a complex model appear to outperform a simple benchmark even when it would not in a real deployment. The paper discusses this explicitly on pp. 12–13. 

Fourth, inappropriate baselines. The paper complains that many papers use weak benchmarks, such as non-seasonal methods on strongly seasonal data, which makes complex methods look better than they really are. 

The deeper statistical interpretation

From a forecasting perspective, the phenomenon usually means one of two things:

Either the series is approximately a martingale / random walk, so the conditional expectation of the future given the past is basically the present value;

or your model has failed to uncover structure beyond lag-1 persistence.

Those two cases can look nearly identical in output space. That is why forecast evaluation is so important.

A useful practical interpretation is:
if your complex model’s forecast increments are close to zero after transforming back to the original scale, then the model is effectively saying “I have no evidence the next move is predictably positive or negative,” which collapses back to the last observed value. On some datasets that is the right answer.

When this happens on seasonal data

There is a seasonal analogue. On seasonal series, a model may end up reproducing the last seasonal observation, which is basically a seasonal naïve forecast. The paper says that on clear seasonal data, the relevant simple benchmark is seasonal naïve, not ordinary naïve. So if a complex model seems strong only because it is rediscovering the seasonal pattern in the simplest possible way, that is not much evidence of added value. 

How I would diagnose the phenomenon in practice

I would ask:

1. Does the series look random-walk-like or strongly integrated?
   If yes, matching naïve may be expected.

2. Does differencing or detrending reveal residual structure?
   If not, there may be no learnable signal.

3. Does the model beat naïve or seasonal naïve under rolling-origin out-of-sample evaluation?
   If not, it is just a more expensive baseline.

4. Is the gain statistically significant across many series or forecast origins?
   The paper recommends significance testing when possible because tiny apparent gains can be noise. 

5. Could leakage be inflating performance?
   This is especially important if preprocessing was done before splitting. 

6. Are the forecast metrics appropriate for the data type?
   The paper spends much of Sections 6–7 on this because wrong metrics can hide the truth. 

What the paper’s overall position is

The paper is not saying “ML is useless for forecasting.” It says almost the opposite: ML and DL can be competitive, but only when forecasting-specific pitfalls are handled correctly. One of the biggest pitfalls is not realizing when a simple benchmark like naïve is already doing nearly all the useful work. In that setting, a complex model that gives the same predictions is not necessarily wrong — it may simply be revealing that the dataset does not support more ambitious forecasting at that horizon. 

The paper’s final guidelines reinforce exactly this: always compare against the simplest valid benchmarks such as naïve and seasonal naïve; do not trust forecast plots alone; avoid leakage; and choose evaluation procedures and error measures carefully. Those are among the paper’s headline conclusions on pp. 58–60. 

My concise read of the phenomenon is:

A complex ML model giving essentially the same forecast as a naïve method usually means one of three things:

* the data are close to random-walk / unit-root behavior and naïve is close to optimal,
* the model has learned only persistence because there is little usable signal,
* or the evaluation/modeling setup is poor and is hiding the fact that the model adds no real skill.

# Why does the paper said "next period equals the last observed value" is a good prediction?

Because in the paper’s example, the series is assumed to behave like a **random walk**:

$$y_{t+1} = y_{t} + \varepsilon_{t}$$

where the next move is just the current value plus **unpredictable noise**. If that is the data-generating process, then the future change has mean zero, so the best point forecast from time (t) is simply the current value:

$$\hat{y}_{t+h} = y_{t}$$

That is why the paper says “next period equals the last observed value” can be a good prediction. It is not because it is clever-looking; it is because, under that process, there is **no extra predictable structure to exploit** beyond the last observation. 

The paper is using “good” in a forecasting sense: **best possible given the information available now**. It explicitly says that for a series with unit-root non-stationarity and no further predictable properties, the naïve forecast is the **theoretically best forecast**, and more complex methods have no true predictive power beyond it. Any apparent improvement would just be chance on a finite sample. 

A second reason is evaluation. The paper stresses that a forecast should be judged by **error minimization**, not by whether it looks realistic on a chart. It even says that a fixed-origin naïve forecast can look unrealistic because it is flat, but it may still be the **best possible forecast in the sense that it minimizes error based on the information available at present**. 

So the logic is:

* if the series is basically today’s value plus random shock,
* and the shock is not forecastable,
* then the safest and optimal point prediction is **no change**.

That is why the paper treats the naïve forecast as a serious benchmark, especially for integrated or random-walk-like series. It also says that on such series, complex models often end up following the series in much the same way as the naïve forecast, with only small or spurious apparent gains. 

In plain language: the paper is saying **a good forecast is not the one that seems sophisticated; it is the one that best uses the available information**. When the future movement is mostly unpredictable noise, “tomorrow ≈ today” is the right answer.

For your aluminum example, this means: if monthly LME prices are close to a random-walk-like process, then a model that mostly says “next month will be near this month” may be doing exactly what the paper considers reasonable. The real question is whether it **beats the naïve benchmark out of sample**, not whether it draws a smoother line.

# Does the paper explain how to make "unpredictable noise" forecastable?

Not really — **not in the sense of turning true white noise into something forecastable**.

The paper’s position is basically: if the series is a random walk and the innovation is white noise with no further predictable structure, then the naïve forecast is theoretically best, and more complex models have **no true predictive power beyond it**. So if the “noise” is genuinely unpredictable, the paper is saying you do **not** make that part forecastable from the past of the same series.  

What the paper **does** explain is something slightly different:

It explains how to handle cases where the problem is not pure noise, but rather **hidden structure plus non-stationarity**. For example, it says forecasting can use not only past values of the target series but also **exogenous variables** available by the forecast origin, or even future values of those variables if they are known. 

It also discusses ways to make the series easier to model by addressing non-stationarity, such as:

* lag-1 differencing,
* log transforms for exponential trends,
* STL decomposition,
* moving-average smoothing,
* percentage-change transforms,
* wavelet transforms.

And it says that deterministic trend or seasonality can sometimes be **extracted and predicted separately**, leaving a more stationary remainder for the model. 

But that is **not** the same as making white noise forecastable. It is really:

* remove or isolate predictable components,
* add informative inputs,
* then model what remains.

The paper is also explicit that some changes are simply not addressable. For example, with abrupt structural breaks in the unknown future, it says it can be **impossible** for models to handle that change, and no evaluation strategy fixes it. 

So the clean answer is:

**No, the paper does not explain how to make truly unpredictable noise forecastable.**
It explains how to check whether there is any predictable structure at all, and how to expose that structure through preprocessing, decomposition, benchmarking, and sometimes external variables. If after that the remainder is still just noise, then the paper’s message is that you should accept that and compare against the naïve benchmark rather than expect ML to create signal from nothing.

For your aluminum case, that means the right question is probably not “how do I make the noise forecastable?” but “is there any real signal beyond persistence, and do external drivers or transformations reveal it?”

# How do i know if the time series dataset has "unpredictable noise" or "hidden structure plus non-stationarity"?

Yes. For a financial series like **LME aluminum**, the right way to think about this is not “is the series predictable or not?” but **which part is predictable**: the **price level**, the **return**, or the **volatility/regime**. The literature on commodities and LME metals suggests a common pattern: **levels are often close to a stochastic trend and hard to beat with a no-change or futures-based benchmark, while volatility and some state-dependent return structure are still forecastable.** ([MDPI][1])

Here is the most useful distinction.

**“Unpredictable noise”** does **not** mean the whole dataset is meaningless. In finance it usually means the **conditional mean** of the next return is very hard to predict with the information set you have, so a more complex mean model does not reliably beat a random walk, zero-return, or futures-based forecast out of sample. But that same series can still show **predictable volatility**, because ARCH/GARCH-type dependence and long-memory volatility are a different kind of structure from mean predictability. That distinction is foundational in the ARCH/GARCH and realized-volatility literature, and it has been documented specifically for LME metals. ([ScienceDirect][2])

**“Hidden structure plus non-stationarity”** means the raw price series looks hard to model because it mixes together a stochastic trend, breaks, regimes, volatility clustering, or cointegrating relationships. In that case, the raw level may look random-walk-like, but after differencing, adjusting for breaks, conditioning on exogenous predictors, or modeling volatility separately, you still find structure left in the data. The literature on unit roots and structural breaks treats this as a central issue, because standard unit-root conclusions can change once breaks are handled explicitly. ([MDPI][3])

So the practical question for your aluminum series is:

**Does the series remain unpredictable after you control for the obvious sources of non-stationarity and hidden dependence?**

A good diagnostic workflow for **LME aluminum** looks like this.

First, **separate level, return, and volatility**. For a monthly series, define log price $p_t$, monthly return $r_t = \Delta p_t$, and a volatility proxy such as $|r_t|, r_t^2$, or realized volatility if you have higher-frequency data. This matters because commodity papers repeatedly find that forecasting **nominal price levels** is much harder than forecasting some aspects of volatility or state variables, and aluminum-specific studies continue to treat the forecasting problem as difficult even with richer predictor sets. ([MDPI][4])

Second, ask whether the **price level behaves like a random walk or near-unit-root process**. A sensible battery is to use a unit-root test such as Phillips–Perron or an ADF-type test, **together with** KPSS, because KPSS reverses the null and tests stationarity rather than a unit root. Then add a **break-aware** test such as Zivot–Andrews, because breaks can distort ordinary unit-root inference, and a **variance-ratio** test, because under a random-walk null the variance ratio should be near one. Also be cautious about over-interpreting unit-root p-values, since the literature stresses that unit-root tests often have low power. 

How to read that battery:

* If log price looks unit-root-like under PP/ADF-type testing, KPSS rejects stationarity, break-aware tests do not rescue stationarity, and the variance ratio stays near one, then your **level** is behaving like a stochastic trend.
* If ordinary unit-root tests point to a unit root but break-aware tests show important endogenous breaks, or the conclusion changes materially once breaks are modeled, then you likely have **non-stationarity with hidden structure**, not just pure noise. ([JSTOR][5])

Third, move to **returns**, because that is where you test for residual mean structure. Fit a plain benchmark on returns first, then ask whether returns or residuals still show dependence. The classic diagnostics here are serial-correlation checks on returns, squared-residual autocorrelation tests such as McLeod–Li, and then a **BDS test** on residuals from your mean or mean-volatility model. McLeod–Li is useful because dependence often survives in squared residuals even when raw returns look nearly uncorrelated, and the BDS test is designed to detect remaining serial dependence and model misspecification in residuals. ([IDEAS/RePEc][6])

Interpretation here is straightforward:

* If monthly returns show little serial dependence, squared returns also look clean, and BDS on standardized residuals does not reject, your **mean process** is close to noise.
* If returns have weak linear dependence but squared returns are clearly dependent, or BDS still rejects after a simple AR model, then the series is **not iid noise**; the hidden structure is probably nonlinear or in the conditional variance. ([IDEAS/RePEc][6])

Fourth, test **volatility predictability** separately. This is where many financial series reveal structure even when the mean does not. ARCH/GARCH models exist precisely because conditional variance can depend on past squared shocks and past variance, and realized-volatility work shows that multi-horizon volatility components are forecastable. For LME non-ferrous metals, realized volatility has been shown to be well captured by rolling HAR-GARCH-type models, and aluminum/copper volatilities have been found to display long-memory commonality. ([ScienceDirect][2])

For **LME aluminum specifically**, that is a major clue: even if the **next-month level** is hard to predict, the **volatility state** is usually not just noise. Studies on LME metals report long-memory volatility, common volatility factors, and better fit from HAR-GARCH-style models for realized volatility. That means your dataset may be “mean-noisy but volatility-structured,” which is very different from “everything is unpredictable.” ([e-Archivo][7])

Fifth, look for **structural breaks, regime changes, and explosive episodes**. Breaks matter because they can masquerade as persistence, and bubble episodes can make a series look nonstationary for reasons that are not just random walk behavior. For structural breaks, Bai–Perron-style multiple-break procedures are standard. For explosivity, GSADF-style bubble tests are the workhorse, and recent aluminum research applies GSADF directly to world aluminum prices. There is also non-ferrous-metal evidence of multiple structural break points and elevated volatility risk around crisis periods. ([EconPapers][8])

Sixth, test whether there is **economic structure** that a univariate price model misses. On LME metals, the evidence is not “nothing predicts anything.” Rather, the evidence is **selective predictability** from the right state variables. Research on LME metals finds that **financial variables, proxies for global economic activity, and the basis** can predict both futures and spot returns. Aluminum-specific work finds predictive content in **commodity currencies** for both spot and futures aluminum prices, and another line of work finds predictive content in **convenience yields** and detrended oil price for mineral spot prices, including aluminum. There is also structural evidence that electricity prices matter for aluminum economics, though the pass-through to equilibrium aluminum prices is incomplete. ([ScienceDirect][9])

That gives you a very concrete predictor shortlist for monthly LME aluminum:
**basis / interest-adjusted basis, convenience yield, commodity currencies, global-activity proxies, oil or energy variables, electricity-related cost proxies, and cross-metal information such as copper.** Those are not arbitrary ML features; they are the variables the aluminum and commodity literature says are economically connected to the price process. ([ScienceDirect][10])

Seventh, judge everything by **strict out-of-sample comparisons**, not in-sample fit. For commodity prices, the literature argues that the random walk remains a crucial benchmark, but at short horizons it should usually be **supplemented by a futures-based forecast**. That is especially relevant for storable commodities, because spot and futures are linked by arbitrage, storage costs, and convenience yield. In the Federal Reserve study, futures generally beat a random walk only modestly on average, but by more when the spot-futures gap is large. That is exactly why aluminum benchmarking should not stop at “naïve versus Prophet”; it should include the matched-horizon futures forecast too. ([MDPI][1])

For the out-of-sample testing itself, use **Clark–West** when your richer model nests a simple benchmark, and **Giacomini–White** when you want a rolling, real-time-style comparison of predictive ability. Clark–West is built around the fact that larger nested models mechanically add estimation noise, so raw MSPE comparisons are biased against the parsimonious benchmark under the null. ([ScienceDirect][11])

Putting all of this together, my best research-based prior for **monthly LME aluminum** is:

* **Price level:** probably close to a stochastic trend and hard to beat consistently with a pure univariate model. Random walk and futures-based forecasts are serious baselines. ([MDPI][1])
* **Return mean:** not obviously forecastable all the time, but not proven to be pure noise either; there is evidence that basis, financial variables, global activity, and commodity currencies can help. ([ScienceDirect][9])
* **Volatility/regime:** definitely not pure noise; LME aluminum shows long-memory volatility, commonality with other metals, crisis-related regime changes, and possibly explosive episodes. ([e-Archivo][7])

So the most likely answer for your dataset is **not** “it is pure unpredictable noise” and also **not** “it has a clean stable structure in the raw level.” The more accurate description is:

**LME aluminum is often close to random-walk-like in the level, but it still contains hidden structure in volatility, regimes, and some conditional predictors.**

A practical decision rule is this:

Call it **mostly unpredictable noise in the mean** only if, after transforming to returns, handling breaks, and testing exogenous predictors, you still cannot beat **both** the random-walk and the futures-based benchmark out of sample, and the residual diagnostics stop finding dependence. ([MDPI][1])

Call it **hidden structure plus non-stationarity** if the raw level looks unit-root-like, but one or more of these hold: volatility models improve forecast accuracy, break or bubble tests flag regime changes, or economically motivated predictors such as basis, convenience yield, global activity, or commodity currencies deliver incremental out-of-sample value. ([e-Archivo][7])

For your **current Prophet setup**, the main danger is that Prophet on raw prices can look good because it tracks the stochastic trend, while adding little genuine signal. The fastest way to find out is to rebuild the exercise as a **three-track evaluation**: one track for next-month **price level**, one for next-month **return sign/size**, and one for next-month **volatility**—each compared against **random walk** and **futures-based** benchmarks. That will tell you whether your aluminum dataset is “noise,” “hidden structure,” or the far more common hybrid: **mean almost noise, volatility structured.** ([MDPI][1])

[1]: https://www.mdpi.com/2571-9394/3/2/27 "Forecasting Commodity Prices: Looking for a Benchmark | MDPI"
[2]: https://www.sciencedirect.com/science/article/pii/0304407686900631 "Generalized autoregressive conditional heteroskedasticity - ScienceDirect"
[3]: https://www.mdpi.com/2225-1146/5/2/22?utm_source=chatgpt.com "Unit Roots and Structural Breaks"
[4]: https://www.mdpi.com/2813-0324/11/1/13 "The Forecasting of Aluminum Prices: A True Challenge for Econometric Models | MDPI"
[5]: https://www.jstor.org/stable/1391541?utm_source=chatgpt.com "Further Evidence on the Great Crash, the Oil-Price Shock, ..."
[6]: https://ideas.repec.org/a/bla/jtsera/v4y1983i4p269-273.html "Diagnostic Checking Arma Time Series Models Using Squared‐Residual Autocorrelations"
[7]: https://e-archivo.uc3m.es/entities/publication/63997c6f-7b20-4cfc-9d33-f2b0186415e9 "Commonality in the LME aluminium and copper volatility processes through a Figarch lens"
[8]: https://econpapers.repec.org/RePEc%3Ajae%3Ajapmet%3Av%3A18%3Ay%3A2003%3Ai%3A1%3Ap%3A1-22?utm_source=chatgpt.com "Computation and analysis of multiple structural change ..."
[9]: https://www.sciencedirect.com/science/article/pii/S2405851321000210 "Do the basis and other predictors of futures return also predict spot return with the same signs and magnitudes? Evidence from the LME - ScienceDirect"
[10]: https://www.sciencedirect.com/science/article/abs/pii/S0301420719305252 "The predictive power of convenience yields - ScienceDirect"
[11]: https://www.sciencedirect.com/science/article/abs/pii/S0304407606000960 "Approximately normal tests for equal predictive accuracy in nested models - ScienceDirect"

# What if the data is daily?

Yes — **daily data changes the plan materially**.

For **daily LME aluminum**, the right question is no longer just “can I forecast the price?” but **which layer is forecastable**: the **price level**, the **next-day return**, the **volatility**, or the **regime/break state**. The research picture for LME and industrial metals points to a hybrid answer: **daily aluminum levels are often close to random-walk-like, mean returns are at best weakly and episodically predictable, but volatility, structural breaks, and some economically motivated predictors are clearly not just noise.** ([IDEAS/RePEc][1])

That means my change of plan is: **I would not treat daily LME aluminum primarily as a Prophet-style trend/seasonality problem.** I would split it into three forecasting problems:
(1) **next-day price level** benchmarked against **no-change / random walk**,
(2) **next-day return** benchmarked against **zero return / AR(1)**, and
(3) **next-day volatility** benchmarked against **simple GARCH/HAR-style volatility models**. The literature on non-ferrous metals and LME volatility strongly supports that decomposition. ([IDEAS/RePEc][1])

## My bottom-line prior for daily LME aluminum

If I had to bet **before seeing your data diagnostics**, I would bet on this:

* **Raw daily prices:** mostly a **stochastic trend / near-unit-root** object.
* **Daily mean returns:** often close to **noise**, but not always; some predictors can help.
* **Daily volatility:** **definitely structured**, with clustering and long memory.
* **Regimes / breaks / bubbles:** important enough that ignoring them can make the series look “more random” than it really is. ([IDEAS/RePEc][1])

So the most likely answer is **not** “pure unpredictable noise” and **not** “a stable predictable signal in raw prices.” It is usually:

> **near-random-walk in the level, weak or unstable predictability in mean returns, and substantial predictability in volatility/regimes.** ([IDEAS/RePEc][1])

## Why daily data changes the diagnosis

At daily frequency, financial and commodity series usually become **harder to predict in the conditional mean** and **easier to detect in the conditional variance**. For LME metals specifically, studies find that aluminum and related non-ferrous metal volatilities exhibit **long memory**, **commonality**, and are well captured by **HAR-GARCH / FIGARCH-type models** rather than by a flat-noise assumption. ([IDEAS/RePEc][1])

There is also evidence that **daily LME structure matters**. One daily-data study on the LME used **daily non-overlapping observations from 2000 to 2016** and concluded that the market was not efficient in that framework, while another line of work has documented **daily seasonality** in LME futures behavior. That does **not** mean easy profits, but it does mean “daily = pure iid noise” is too strong a prior. ([MDPI][2])

LME market structure matters here too: the exchange provides **cash and rolling 3-month futures**, and official prices are tied to specific trading sessions. That means for aluminum, the **basis / futures-spot relationship** is not some optional feature engineering trick — it is part of the market’s native information structure. ([MDPI][2])

## How to tell “unpredictable noise” from “hidden structure + non-stationarity”

For **daily LME aluminum**, I would use this decision framework.

### 1. Test the **log price level** first, not just the raw price

$Let (p_t = \log P_t)$. Your first task is to determine whether the daily level behaves like a **unit-root / random-walk-like process** or whether that conclusion is being distorted by **breaks**.

A good battery is:

* a **unit-root test** such as ADF/PP,
* a **stationarity test** such as KPSS,
* a **variance-ratio test** for random walk behavior,
* and at least one **break-aware unit-root test** such as **Zivot–Andrews** or the later break-under-null approach of **Kim–Perron**. The reason is important: break-unaware tests can misclassify a **broken-trend stationary process** as a unit root, while break-aware tests explicitly allow the break date to matter. The variance-ratio literature is also directly tied to the random-walk hypothesis. ([Massachusetts Institute of Technology][3])

Interpretation:

* If log prices keep looking unit-root-like **even after break-aware testing**, then the **level** is behaving like a stochastic trend.
* If ordinary tests say “unit root” but break-aware tests materially change the conclusion, then you do **not** have “just noise”; you have **non-stationarity with structure hidden by breaks**. ([University of Glasgow][4])

### 2. Move immediately from levels to **daily returns**

Define daily log return $(r_t = p_t - p_{t-1})$. This is the series you use to diagnose **mean predictability**.

If returns are truly close to unpredictable noise in the mean, then:

* linear autocorrelation should be weak,
* simple AR terms should add little,
* and more complex models should not reliably beat **zero-return** or **no-change in price** out of sample. ([Social Science Computing Core][5])

But you should **not stop at autocorrelation**. Financial series often look uncorrelated in raw returns while still containing **nonlinear dependence**. That is where the **BDS test** and **McLeod–Li squared-residual tests** are useful: BDS is designed as an **independence / iid-type residual check**, and McLeod–Li-type tests use **squared residual autocorrelations** to detect nonlinear dependence missed by linear models. ([SAGE Journals][6])

Interpretation:

* If returns show little linear dependence **and** BDS does not reject iid-like residual behavior after a simple mean model, then your **mean** is close to noise.
* If returns look linearly weak but BDS or squared-residual tests still reject, then the series is **not just noise**; the structure is nonlinear and/or in higher moments. ([SAGE Journals][6])

### 3. Treat **volatility** as a separate forecasting target

This is the part most people miss.

For daily LME aluminum, volatility is where the literature is strongest. Research on LME non-ferrous metals finds that realized volatility is well captured by **rolling HAR-GARCH** models, and aluminum/copper volatility has been found to exhibit **long-memory behavior** and substantial **commonality**. ([IDEAS/RePEc][1])

So if your daily return mean looks unpredictable, that does **not** settle the question. You should still test:

* autocorrelation of $(r_t^2) or (|r_t|)$,
* **ARCH-LM**-type effects,
* **GARCH / FIGARCH / HAR** models,
* and, if you have intraday data, **realized volatility** models. The ARCH/GARCH tradition exists exactly because financial volatility is often conditionally predictable even when returns are not. ([Stern School of Business][7])

This leads to an important practical conclusion:

If your next-day **direction** is hard to forecast but your next-day **variance** is forecastable, your dataset is **not pure unpredictable noise**. It has **hidden structure in risk**, not necessarily in mean return. ([IDEAS/RePEc][1])

### 4. Check for **regimes, breaks, and explosive episodes**

Daily aluminum prices can go through episodes where the process changes materially. Multiple-break methods in the **Bai–Perron** tradition are standard for identifying structural changes, and recent aluminum-specific work has used **GSADF** methods to test for bubble-like explosive periods. More broadly, metals research using daily and weekly data has emphasized **structural breaks, non-linearity, stationarity issues, and bubble incidences** as core properties to test. 

This matters because a regime-shifting series can fool you in two opposite ways:

* a single stable model may look bad and make you conclude “noise,” when the real issue is **unmodeled breaks**;
* or a flexible model may overfit local regimes and look great in-sample, even though it has no stable forecasting edge. 

### 5. Test economically motivated predictors, not just lags

For LME aluminum, there is evidence that **financial variables**, **global activity proxies**, and the **basis** help predict both futures and spot returns, and aluminum-specific work finds predictive content in **commodity currencies** for spot and futures aluminum prices. ([ScienceDirect][8])

So the right multivariate question is not “can ML learn from price lags alone?” but:

> **Does daily aluminum return predictability appear once I condition on the right state variables?**

For aluminum, the most defensible candidate predictors are:

* **basis / term structure variables**,
* **commodity-exporter FX**,
* **global activity / risk variables**,
* and, for slower-moving structure, **inventory / storage / convenience-yield-type information**. Theory-of-storage work on LME base metals links price, volatility, inventory, and convenience yield, while aluminum industry research shows that electricity prices matter for the economics of aluminum even if pass-through is incomplete. ([epge.fgv.br][9])

For **daily** forecasting, I would expect **basis, FX, and risk/global-activity variables** to be more useful than slow structural cost variables like electricity, which tend to matter more at medium horizons. That last sentence is my inference from the mix of results above.

## How to decide formally

Do **rolling out-of-sample** testing, not one holdout split.

For **mean forecasts**, compare:

* **random walk / no-change price**,
* **zero return**,
* **AR(1)**,
* your richer ARX / ML model,
* and, when the horizon matches, a **futures-based benchmark**. Commodity forecasting work from the Federal Reserve found that futures prices generally outperform a random walk only **modestly on average**, but by **more** when the gap between spot and futures is large; broader benchmark work argues that short-horizon commodity forecasting should use **both random-walk and futures-based benchmarks**, not just one. ([Federal Reserve][10])

For significance, use **Clark–West** when the larger model nests the benchmark, and **Giacomini–White** when you want a real-time, rolling predictive-ability comparison. Clark–West is built for the common case where the larger model mechanically adds parameter-estimation noise, and Giacomini–White is designed for realistic out-of-sample predictive-ability testing. ([Social Science Computing Core][5])

## The practical decision rule I would use on your daily data

Call the dataset **“mostly unpredictable noise in the mean”** if all of these happen:

* log prices look unit-root-like even after break-aware testing,
* daily returns show little dependable linear or nonlinear structure,
* richer models do **not** beat **no-change / zero-return** out of sample,
* and any apparent gains vanish under **Clark–West / Giacomini–White** testing. ([Massachusetts Institute of Technology][3])

Call it **“hidden structure plus non-stationarity”** if any of these happen:

* break-aware tests change the inference versus plain unit-root tests,
* BDS or squared-residual tests reject iid-like behavior,
* volatility models materially outperform a flat-variance benchmark,
* or basis / FX / global-activity predictors add out-of-sample forecasting value. ([SAGE Journals][6])

## My most likely conclusion for **daily LME aluminum**

My honest prior is:

**Daily LME aluminum is unlikely to be pure noise.**
But it is also unlikely that the **raw daily level** contains strong stable forecastability that a trend model like Prophet will unlock. The more likely outcome is:

* **level:** near random walk,
* **mean return:** weak and unstable,
* **volatility / regimes / basis-sensitive states:** meaningfully structured. ([IDEAS/RePEc][1])

That is the “deep” answer: for daily aluminum, the right task is usually **not** “predict tomorrow’s price path smoothly,” but **diagnose whether the predictive content lives in mean, variance, or state variables.**

[1]: https://ideas.repec.org/a/eee/ecmode/v51y2015icp1-12.html "The course of realized volatility in the LME non-ferrous metal market"
[2]: https://www.mdpi.com/2227-7072/6/1/32 "Testing Efficiency of the London Metal Exchange: New Evidence | MDPI"
[3]: https://web.mit.edu/Alo/www/Papers/lo-mackinlay-89.html "Abstract: THE SIZE AND POWER OF THE VARIANCE RATIO TEST IN
FINITE SAMPLES: A MONTE CARLO INVESTIGATION"
[4]: https://www.gla.ac.uk/media/Media_219103_smxx.pdf?utm_source=chatgpt.com "Unit Roots and Structural Breaks: A Survey of the Literature"
[5]: https://www.ssc.wisc.edu/~kwest/publications/2000/Approximately%20Normal%20Tests%20for%20Equal%20Predictive%20Accuracy%20in%20Nested%20Models.pdf "doi:10.1016/j.jeconom.2006.05.023"
[6]: https://journals.sagepub.com/doi/full/10.1177/1536867X211025796?utm_source=chatgpt.com "The BDS test of independence - Christopher F. Baum, Stan ..."
[7]: https://www.stern.nyu.edu/rengle/GARCH101.PDF?utm_source=chatgpt.com "An Introduction to the Use of ARCH/GARCH models in ..."
[8]: https://www.sciencedirect.com/science/article/pii/S2405851321000210 "Do the basis and other predictors of futures return also predict spot return with the same signs and magnitudes? Evidence from the LME - ScienceDirect"
[9]: https://epge.fgv.br/conferencias/commodity-prices/files/HelyetteGeman.pdf?utm_source=chatgpt.com "Theory of Storage, Inventory and Volatility in the LME Base ..."
[10]: https://www.federalreserve.gov/pubs/ifdp/2011/1025/ifdp1025.pdf "Evaluating the Forecasting Performance of Commodity Futures Prices"
