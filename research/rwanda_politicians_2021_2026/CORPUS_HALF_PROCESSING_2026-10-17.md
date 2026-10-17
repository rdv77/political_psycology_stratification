# Половина корпуса Руанды: обработка и найденные характеристики

[Основные результаты](CORPUS_HALF_FINDINGS_2026-10-17.md) · [Характеристики на русском](CORPUS_HALF_PROPERTIES_RU_2026-10-17.md) · [Качество](CORPUS_HALF_QUALITY_2026-10-17.md)

Выбрано **12,205 публикаций**. Проверка релевантности, включая ответы review, завершена для **12,205 документов**; извлечение done — для **10,236 документов**.

Учтённый расход: **$69.5248** из $70.00. Неразрешённых запросов: 1; зарезервировано **$0.0610**. Свободный остаток после резерваций: **$0.4142**.

Это сводка извлечённых свидетельств и черновых характеристик, а не итоговые оценки политиков. Полные публикации, цитаты, URL, авторы и ответы модели остаются локально.

## Покрытие обработки

| Показатель | Документов |
|---|---:|
| Выбранная половина | 12205 |
| Есть подготовленные задания отбора | 12205 |
| Отбор пока не подготовлен | 0 |
| Отбор done по всем подготовленным частям | 11925 |
| Отбор содержит review | 280 |
| Отбор содержит error / uncertain / running / pending | 0 / 0 / 0 / 0 |
| Признаны содержательно относящимися к участникам | 11693 |
| Подготовлено извлечение | 11693 |
| Извлечение done по всем подготовленным частям | 10236 |
| Извлечение содержит review / error / uncertain / running / pending | 3 / 1 / 1 / 0 / 1452 |

Статусы разных частей одного документа могут пересекаться. Review обозначает необходимость проверки, error — техническую ошибку, uncertain — неразрешённый исход запроса. Они не характеризуют политическую позицию.

## Расходы по этапам и моделям

Включены все квитанции корпуса, в том числе неактивные версии и неуспешные ответы. Резервации показаны отдельно от учтённой стоимости.

| Этап | Модель | Статус | Запросов | Входных токенов | Выходных токенов | Учтено, $ | Резерв, $ |
|---|---|---|---:|---:|---:|---:|---:|
| describe | gpt-5.1 | done | 8 | 216269 | 21120 | 0.481536 | 0.000000 |
| describe | gpt-5.1 | review | 2 | 53553 | 12000 | 0.186941 | 0.000000 |
| extract | gpt-5.1 | done | 10236 | 19355427 | 3530896 | 59.499788 | 0.000000 |
| extract | gpt-5.1 | error | 1 | 0 | 0 | 0.000000 | 0.000000 |
| extract | gpt-5.1 | pending | 1452 | 0 | 0 | 0.000000 | 0.000000 |
| extract | gpt-5.1 | review | 3 | 17507 | 12288 | 0.144764 | 0.000000 |
| extract | gpt-5.1 | uncertain | 1 | 0 | 0 | 0.000000 | 0.060966 |
| relevance | gpt-4.1-mini | done | 11925 | 17991613 | 1009485 | 8.811821 | 0.000000 |
| relevance | gpt-4.1-mini | review | 280 | 910207 | 22410 | 0.399939 | 0.000000 |

## Проверка качества

Подключена автоматическая структурная проверка: наличие цитат в источнике, координаты, связи с контекстом и исходные решения нормализации. Эти проверки не доказывают правильность смысла, атрибуции или датировки события.

| Проверка | Количество |
|---|---:|
| Принятых фрагментов проверено структурно | 11285 |
| Фрагментов с замечаниями при повторной проверке | 0 |
| Проверено контекстов запросов | 23898 |
| Завершённых извлечений с сохранённым контролем нормализации | 10236 |
| Завершённых извлечений без такой квитанции | 0 |
| Предложено фрагментов до нормализации | 16864 |
| Отсечено нормализацией | 5569 |
| Изменён тип/допуск нормализацией | 434 |
| Сняты неподтверждённые даты | 421 |

Исходные причины нормализации: actor\_not\_local — 4796; attribution\_not\_exact — 213; attribution\_not\_local — 210; context\_attribution\_downgraded — 68; date\_not\_grounded — 16; date\_not\_local — 388; date\_year\_missing — 17; kind\_basis\_mismatch — 37; orphan\_date\_evidence\_cleared — 3065; own\_attribution\_downgraded — 367; quote\_not\_exact — 291; validator\_rejected — 22.

Для дополнительного чтения автоматически подготовлено 178 фрагментов основной выборки, 80 фрагментов повышенного риска и 28 документов для проверки пропусков. Эти наборы могут пересекаться; создание пакета не означает его содержательной проверки. Итоги выполненного чтения источников приводятся в [отдельной сводке качества](CORPUS_HALF_QUALITY_2026-10-17.md). Оценка точности всего корпуса по целевой выборке не рассчитывается.

## Что найдено

Активных фрагментов: **11285**; допустимых свидетельств собственных публичных заявлений/действий: **7039**. Они есть у **35 из 42** участников.

Ниже частоты коротких названий из trait_candidates. Объединены регистр, Unicode и пробелы; отрицания сохранены. Частоты относятся к фрагментам, не независимым событиям и не выраженности установки. Названия не утверждены.

| Кандидат характеристики | Допустимых фрагментов позиции | Участников с такими опорами | Всего фрагментов, включая образ |
|---|---:|---:|---:|
| diplomatic | 508 | 12 | 590 |
| diplomatic\_engagement | 146 | 12 | 172 |
| development\_oriented | 61 | 12 | 112 |
| strategic | 50 | 12 | 94 |
| international\_engagement | 81 | 11 | 101 |
| development-oriented | 59 | 11 | 103 |
| regional\_engagement | 65 | 10 | 69 |
| regional\_cooperation | 45 | 10 | 47 |
| pro-innovation | 44 | 10 | 45 |
| diplomatic engagement | 108 | 8 | 112 |
| development-focused | 51 | 8 | 66 |
| support for international cooperation | 14 | 8 | 14 |
| pro-cooperation | 80 | 7 | 87 |
| pragmatic | 69 | 7 | 84 |
| technocratic | 48 | 7 | 48 |
| pro-investment | 31 | 7 | 33 |
| supports\_international\_cooperation | 15 | 7 | 15 |
| results-oriented | 12 | 7 | 13 |
| collaborative | 9 | 7 | 13 |
| pro-technology | 67 | 6 | 69 |
| coopératif | 54 | 6 | 58 |
| commemorative | 37 | 6 | 39 |
| future-oriented | 30 | 6 | 32 |
| innovation-oriented | 24 | 6 | 25 |
| leadership | 18 | 6 | 36 |
| regional\_integration | 18 | 6 | 19 |
| future\_oriented | 16 | 6 | 17 |
| supportive\_of\_international\_cooperation | 15 | 6 | 17 |
| pro\_innovation | 14 | 6 | 14 |
| pro-digitalization | 13 | 6 | 13 |
| pro-international-cooperation | 11 | 6 | 11 |
| pro\_international\_cooperation | 10 | 6 | 11 |
| public\_engagement | 8 | 6 | 8 |
| security-focused | 126 | 5 | 153 |
| cooperative | 61 | 5 | 74 |
| cautious | 46 | 5 | 47 |
| optimistic | 15 | 5 | 15 |
| pro-development | 11 | 5 | 13 |
| supports\_regional\_integration | 11 | 5 | 15 |
| pro-growth | 10 | 5 | 10 |
| diplomatic\_participation | 9 | 5 | 9 |
| multilateralist | 9 | 5 | 9 |
| reformist | 9 | 5 | 10 |
| regional\_diplomacy | 79 | 4 | 85 |
| defensive | 45 | 4 | 54 |
| economic\_development\_focus | 19 | 4 | 20 |
| pro-foreign-investment | 16 | 4 | 18 |
| multilateral\_engagement | 13 | 4 | 13 |
| regional engagement | 13 | 4 | 13 |
| pro-international\_cooperation | 12 | 4 | 14 |
| development\_focus | 11 | 4 | 20 |
| economic\_diplomacy | 11 | 4 | 11 |
| international\_cooperation | 11 | 4 | 15 |
| public\_health\_concern | 11 | 4 | 12 |
| ceremonial\_role | 9 | 4 | 9 |
| supportive\_of\_bilateral\_relations | 9 | 4 | 10 |
| economic cooperation focus | 8 | 4 | 10 |
| engages\_in\_diplomacy | 8 | 4 | 8 |
| coopération internationale | 7 | 4 | 7 |
| focus on implementation | 7 | 4 | 7 |

Показаны первые 60 из 12568 буквальных названий; полный частотный список находится в JSON.

## Черновой словарь этапа describe

Группы ниже связаны только буквальными совпадениями названий/заявленных aliases. Это не новые факторы и не автоматическое утверждение семантической эквивалентности. Различающиеся определения требуют проверки.

| Название и варианты | Черновиков | Опорных фрагментов | Допустимых опор позиции | Участников | Различаются определения |
|---|---:|---:|---:|---:|---|
| engagement-in-multilateralism-and-partnerships; Support for international and regional cooperation; support-for-international-and-regional-cooperation | 1 | 8 | 8 | 5 | нет |
| regional and bilateral cooperation on governance and security; Support for cross-border governance and security cooperation; support for international partnerships | 1 | 4 | 4 | 4 | нет |
| pro-ai-development; Support for AI and digital development; support-for-ai-and-digital-skills | 1 | 4 | 4 | 3 | нет |
| Support for digital technology and innovation for development; support\_for\_digital\_innovation; support\_for\_technology\_as\_development\_driver | 1 | 4 | 4 | 3 | нет |
| attention-to-women-and-rural-communities; Support for gender-responsive and rural-inclusive policies; support-for-gender-responsive-and-inclusive-policies | 1 | 3 | 3 | 3 | нет |
| emphasis-on-professional-and-ethical-conduct; Promotion of ethical and disciplined public service; promotion-of-discipline-and-values-in-public-service | 1 | 3 | 3 | 3 | нет |
| Support for youth entrepreneurship and creative/sport sectors; support\_for\_creative\_and\_sports\_industries; support\_for\_youth\_entrepreneurship | 1 | 8 | 8 | 2 | нет |
| commitment\_to\_refugee\_protection; protection-and-assistance-for-refugees; Support for refugee protection and assistance; support-for-refugee-protection; support\_for\_refugee\_assistance | 2 | 5 | 5 | 2 | да |
| development‑oriented use of innovation and entrepreneurship; Support for innovation and entrepreneurial development; support for innovation‑led growth | 1 | 4 | 4 | 2 | нет |
| legalistic\_approach\_to\_disputes; legalistic\_approach\_to\_political\_disputes; support\_for\_international\_legal\_redress | 1 | 4 | 4 | 2 | нет |
| Support for sport and cultural diplomacy; support-for-sport-and-cultural-diplomacy; use-of-sport-for-development-and-image | 1 | 4 | 4 | 2 | нет |
| Pan-regional development and identity framing; pan-regional-framing-of-initiatives; support-for-pan-regional-development-and-identity | 1 | 3 | 3 | 2 | нет |
| preference\_for\_negotiated\_solutions; support\_for\_diplomatic\_conflict\_resolution; support\_for\_diplomatic\_solutions | 1 | 3 | 3 | 2 | нет |
| regional integration orientation; Support for regional and South-South economic cooperation; support for regional or South-South economic cooperation | 1 | 3 | 3 | 2 | нет |
| Advocacy for accountability for genocide and mass atrocities; demand\_for\_genocide\_justice; support\_for\_genocide\_accountability | 1 | 2 | 2 | 2 | нет |
| advocacy\_of\_cross\_border\_cohesion; Regional community and integration discourse; support\_for\_regional\_integration | 1 | 2 | 2 | 2 | нет |
| ai-driven-governance; ai-for-public-systems; Support for AI in public systems | 1 | 2 | 2 | 2 | нет |
| anti-genocide-ideology-prioritisation; opposition-to-genocide-ideology; Priority to combat mass-atrocity ideology | 1 | 2 | 2 | 2 | нет |
| commemorative-engagement; Support for victim remembrance; support-for-victim-remembrance | 1 | 2 | 2 | 2 | нет |
| professionalized-agriculture-support; science-based-farming; Support for scientific and professional agriculture | 1 | 2 | 2 | 2 | нет |
| Support for youth economic opportunity; support-for-young-entrepreneurs; youth-economic-empowerment | 1 | 2 | 2 | 2 | нет |
| pro‑technology‑driven development; support for digital economy; Support for digital transformation | 1 | 5 | 5 | 1 | нет |
| backing-innovative-rural-finance; Support for innovative rural and smallholder finance; support-for-women-and-rural-producers-in-finance | 1 | 4 | 4 | 1 | нет |
| health\_system\_strengthening\_focus; Support for public health investment and system strengthening; support\_for\_public\_health\_investment | 1 | 4 | 4 | 1 | нет |
| alarm\_about\_health\_risks; public\_health\_risk\_alarmism; support\_for\_public\_health\_protection | 1 | 3 | 3 | 1 | нет |
| Critique of global financial architecture; critique of global financial risk assessment; support for alternative rating institutions | 1 | 3 | 3 | 1 | нет |
| epidemic\_preparedness\_advocacy; Public health preparedness and prevention advocacy; public\_health\_risk\_prevention | 1 | 3 | 3 | 1 | нет |
| promotion of AI investment; support for responsible AI deployment in public services; Support for responsible AI investment and use | 1 | 3 | 3 | 1 | нет |
| promotion of frontier technologies for development; Support for innovation and frontier technologies in development; support for innovation-led economic strategy | 1 | 3 | 3 | 1 | нет |
| Recognition of complex security threats and support for security institutions; recognition\_of\_complex\_threats; support\_for\_international\_security\_engagement | 1 | 3 | 3 | 1 | нет |
| anti\_divisionism\_discourse; Promotion of national unity and prevention of renewed division; promotion\_of\_national\_unity | 1 | 2 | 2 | 1 | нет |
| backing advanced medical infrastructure; Support for expanding public healthcare capacity; support for public healthcare expansion | 1 | 2 | 2 | 1 | нет |
| Collaborative technology and AI governance discourse; promotion\_of\_technology\_and\_ai\_collaboration; support\_for\_multilateral\_tech\_rules | 1 | 2 | 2 | 1 | нет |
| focus\_on\_rural\_implementation; Support for implemented climate adaptation finance; support\_for\_climate\_adaptation\_finance | 1 | 2 | 2 | 1 | нет |
| promotion of healing after atrocities; support for reconciliation after mass violence; Support for reconciliation and healing after mass violence | 1 | 2 | 2 | 1 | нет |
| stance on acknowledging past mass crimes; support for post-conflict truth-telling; Support for truth and accountability about past atrocities | 1 | 2 | 2 | 1 | нет |
| stance on judicial safeguards; Support for constitutional and judicial safeguards; support for independent or rights-protective justice system | 1 | 2 | 2 | 1 | нет |
| Advocacy for value‑driven leadership and the public interest; emphasis\_on\_public\_interest\_values; support\_for\_value\_driven\_leadership | 1 | 1 | 1 | 1 | нет |
| Advocacy for youth economic empowerment and investment; encouragement\_of\_youth\_investment; support\_for\_youth\_economic\_empowerment | 1 | 1 | 1 | 1 | нет |
| Advocacy of public‑private partnerships for development finance; pro\_public\_private\_partnerships; support\_for\_private\_capital\_in\_development | 1 | 1 | 1 | 1 | нет |
| backing long-term electricity expansion; Support for advanced energy technologies in electricity strategy; support for advanced or nuclear energy technologies | 1 | 1 | 1 | 1 | нет |
| backing-religious-inclusion-in-national-life; Support for inclusive religious recognition; support-for-inclusive-religious-recognition | 1 | 1 | 1 | 1 | нет |
| climate-adaptation-finance-prioritization; pro-rural-climate-investment; Support for climate adaptation finance | 1 | 1 | 1 | 1 | нет |
| commitment to gain power via elections; positions on democratic electoral competition; Pro‑electoral path to power | 1 | 1 | 1 | 1 | нет |
| Commitment to non-violent political competition; preference for non-violent competition; rejection of political violence | 1 | 1 | 1 | 1 | нет |
| Concern about health-surveillance capacity; public-health-system-critique; regional-health-surveillance-concern | 1 | 1 | 1 | 1 | нет |
| Concern for citizens affected by human trafficking; concern-for-citizens-affected-by-trafficking; opposition-to-human-trafficking | 1 | 1 | 1 | 1 | нет |
| concern\_for\_global\_inclusion\_in\_tech\_governance; support\_for\_inclusive\_global\_tech\_governance; support\_for\_inclusive\_or\_digital\_governance | 1 | 1 | 1 | 1 | нет |
| Constitutional and legal contestation; defence\_of\_constitutional\_rights; support\_for\_rule\_of\_law | 1 | 1 | 1 | 1 | нет |
| Criticism of external failures in atrocity prevention; criticism of international actors over past violence; focus on external failures in atrocity prevention | 1 | 1 | 1 | 1 | нет |
| criticism of supranational or intergovernmental organisations' democratic practice; Critique of intergovernmental organisation governance | 1 | 1 | 1 | 1 | нет |
| Emphasis on self-reliance over external dependency; emphasis on self‑reliance; skepticism toward external saviours | 1 | 1 | 1 | 1 | нет |
| Endorsement of gender equality and women’s leadership; support\_for\_gender\_equality; support\_for\_women\_leadership | 1 | 1 | 1 | 1 | нет |
| environmental-risk-advocacy; industrial-pollution-critique; Public environmental concern | 1 | 1 | 1 | 1 | нет |
| Health accountability orientation; maternal-health-accountability; rights-based-accountability-in-health | 1 | 1 | 1 | 1 | нет |
| linking climate risks to security; support for climate action and vulnerability reduction; Support for climate vulnerability reduction | 1 | 1 | 1 | 1 | нет |
| opposition to incitement to mass violence; rejection of genocidal rhetoric; Rejection of mass-violence incitement | 1 | 1 | 1 | 1 | нет |
| Promotion of economic self‑reliance for refugees; promotion\_of\_economic\_integration\_of\_displaced\_persons; support\_for\_refugee\_self\_reliance | 1 | 1 | 1 | 1 | нет |
| refugee-return-framing; Support for voluntary refugee return; voluntary-repatriation-advocacy | 1 | 1 | 1 | 1 | нет |
| rights-claiming-through-courts; support-for-political-and-civil-rights-litigation; Use of litigation to defend political and civil rights | 1 | 1 | 1 | 1 | нет |
| support for constitutional and institutional stability; Support for constitutional stability and decentralisation; support for decentralised governance | 1 | 1 | 1 | 1 | нет |
| support for judicial independence and due process; Support for judicial safeguards; support for rule‑of‑law safeguards | 1 | 1 | 1 | 1 | нет |

## Все участники

Порядок сохранён из утверждённого справочника. В каждой карточке сначала указаны все извлечённые фрагменты, затем допустимые опоры публичной позиции. Образ, приписанный другими, не подменяет собственную позицию.

### Paul Kagame

Фрагментов: **6454**, допустимых опор позиции: **3461**. Публикаций с опорами позиции: 2393; доменов: 694. Фрагментов без явной даты события: 6235.

Кандидаты по допустимым опорам позиции: diplomatic — 345 фр.; security-focused — 85 фр.; diplomatic\_engagement — 79 фр.; conflict\_resolution — 77 фр.; regional\_diplomacy — 69 фр.; diplomatic engagement — 53 фр.; pro-cooperation — 51 фр.; international\_engagement — 46 фр..

### Edouard Ngirente

Фрагментов: **17**, допустимых опор позиции: **9**. Публикаций с опорами позиции: 9; доменов: 8. Фрагментов без явной даты события: 17.

Кандидаты по допустимым опорам позиции: long\_term\_planning — 2 фр.; pro-nuclear\_energy — 2 фр.; cautious — 1 фр.; commitment to efficiency — 1 фр.; consensus-seeking — 1 фр.; development planning — 1 фр.; development\_oriented — 1 фр.; diplomatic\_participation — 1 фр..

### Justin Nsengiyumva

Фрагментов: **473**, допустимых опор позиции: **379**. Публикаций с опорами позиции: 319; доменов: 97. Фрагментов без явной даты события: 447.

Кандидаты по допустимым опорам позиции: diplomatic — 23 фр.; development-focused — 10 фр.; diplomatic\_engagement — 8 фр.; technocratic — 8 фр.; development-oriented — 6 фр.; regional\_engagement — 6 фр.; pro-innovation — 5 фр.; pro-investment — 5 фр..

### Vincent Biruta

Фрагментов: **122**, допустимых опор позиции: **103**. Публикаций с опорами позиции: 93; доменов: 51. Фрагментов без явной даты события: 116.

Кандидаты по допустимым опорам позиции: diplomatic — 10 фр.; diplomatic\_engagement — 7 фр.; security-focused — 4 фр.; cooperative — 3 фр.; diplomatic engagement — 3 фр.; international\_engagement — 3 фр.; agreement-oriented — 2 фр.; emphasizes implementation of agreements — 2 фр..

### Olivier J. P. Nduhungirehe

Фрагментов: **1658**, допустимых опор позиции: **1386**. Публикаций с опорами позиции: 1237; доменов: 354. Фрагментов без явной даты события: 1552.

Кандидаты по допустимым опорам позиции: diplomatic — 103 фр.; diplomatic engagement — 42 фр.; critical\_of\_DRC\_government — 39 фр.; diplomatic\_engagement — 38 фр.; security-focused — 31 фр.; pro-cooperation — 20 фр.; coopératif — 16 фр.; critical\_of\_Tshisekedi — 15 фр..

### James Kabarebe

Фрагментов: **186**, допустимых опор позиции: **45**. Публикаций с опорами позиции: 42; доменов: 32. Фрагментов без явной даты события: 182.

Кандидаты по допустимым опорам позиции: military\_leadership — 4 фр.; sanctioned\_individual — 4 фр.; sanctioned — 2 фр.; conflict\_mediation\_role — 2 фр.; influence\_régionale — 2 фр.; involvement in regional conflict — 2 фр.; sanctioned\_by\_us — 2 фр.; subject of international sanctions — 2 фр..

### Juvenal Marizamunda

Фрагментов: **118**, допустимых опор позиции: **77**. Публикаций с опорами позиции: 68; доменов: 35. Фрагментов без явной даты события: 112.

Кандидаты по допустимым опорам позиции: diplomatic — 9 фр.; security-focused — 5 фр.; leadership — 4 фр.; ceremonial\_role — 3 фр.; cooperative — 3 фр.; regional\_engagement — 3 фр.; supportive\_of\_regional\_cooperation — 3 фр.; supports regional military cooperation — 3 фр..

### Albert Murasira

Фрагментов: **39**, допустимых опор позиции: **33**. Публикаций с опорами позиции: 26; доменов: 16. Фрагментов без явной даты события: 35.

Кандидаты по допустимым опорам позиции: supports\_refugee\_self\_reliance — 3 фр.; humanitarian\_concern — 2 фр.; pro-refugee\_policy — 2 фр.; supportive\_of\_international\_cooperation — 2 фр.; supports\_voluntary\_repatriation — 2 фр.; advocates\_international\_cooperation — 1 фр.; advocates\_reduction\_of\_aid\_dependence — 1 фр.; commitment to humanitarian assistance — 1 фр..

### Emmanuel Ugirashebuja

Фрагментов: **186**, допустимых опор позиции: **167**. Публикаций с опорами позиции: 153; доменов: 109. Фрагментов без явной даты события: 181.

Кандидаты по допустимым опорам позиции: defensive\_of\_Rwanda\_interests — 27 фр.; critical\_of\_UK\_government — 26 фр.; legalistic — 26 фр.; assertive — 15 фр.; legalistic\_argumentation — 11 фр.; defensive — 9 фр.; legal\_representative — 7 фр.; defensive\_of\_national\_interests — 6 фр..

### Yusuf Murangwa

Фрагментов: **315**, допустимых опор позиции: **241**. Публикаций с опорами позиции: 203; доменов: 75. Фрагментов без явной даты события: 304.

Кандидаты по допустимым опорам позиции: technocratic — 24 фр.; cautious — 9 фр.; concerned\_about\_food\_security — 7 фр.; risk-aware — 6 фр.; economic\_management — 5 фр.; pro-investment — 5 фр.; strategic — 5 фр.; supports inclusive economic development — 5 фр..

### Uzziel Ndagijimana

Фрагментов: **11**, допустимых опор позиции: **10**. Публикаций с опорами позиции: 8; доменов: 4. Фрагментов без явной даты события: 11.

Кандидаты по допустимым опорам позиции: commemorative\_attitude — 1 фр.; community\_support\_orientation — 1 фр.; emphasis on asset quality — 1 фр.; emphasis on national development — 1 фр.; emphasis on youth responsibility — 1 фр.; engagement with capital markets — 1 фр.; focus on profitability drivers — 1 фр.; interest in sustainable economic growth — 1 фр..

### Paula Ingabire

Фрагментов: **359**, допустимых опор позиции: **288**. Публикаций с опорами позиции: 226; доменов: 140. Фрагментов без явной даты события: 357.

Кандидаты по допустимым опорам позиции: pro-technology — 42 фр.; pro-innovation — 18 фр.; future-oriented — 11 фр.; innovation-oriented — 9 фр.; education-focused — 7 фр.; development-oriented — 6 фр.; pro-AI — 6 фр.; technocratic — 6 фр..

### Sabin Nsanzimana

Фрагментов: **236**, допустимых опор позиции: **218**. Публикаций с опорами позиции: 192; доменов: 101. Фрагментов без явной даты события: 233.

Кандидаты по допустимым опорам позиции: concerned\_about\_public\_health — 8 фр.; public\_health\_concern — 8 фр.; innovation-oriented — 5 фр.; concern for maternal health — 4 фр.; concern for public health — 4 фр.; concern\_for\_public\_health — 4 фр.; prioritizes\_infectious\_disease\_control — 4 фр.; focus on accountability — 3 фр..

### Judith Uwizeye

Фрагментов: **4**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 4.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Inès Mpambara

Фрагментов: **1**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 1.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Valentine Uwamariya

Фрагментов: **16**, допустимых опор позиции: **11**. Публикаций с опорами позиции: 11; доменов: 11. Фрагментов без явной даты события: 15.

Кандидаты по допустимым опорам позиции: concern\_for\_living\_conditions — 3 фр.; concern about housing conditions — 1 фр.; concern about sanitation — 1 фр.; concern for community protection — 1 фр.; concern for living conditions — 1 фр.; critical\_of\_poor\_infrastructure — 1 фр.; critical\_of\_sanitation\_and\_infrastructure — 1 фр.; criticism of local infrastructure — 1 фр..

### Jeanne d'Arc Mujawamariya

Фрагментов: **1**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 1.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Gérardine Mukeshimana

Фрагментов: **66**, допустимых опор позиции: **63**. Публикаций с опорами позиции: 48; доменов: 33. Фрагментов без явной даты события: 66.

Кандидаты по допустимым опорам позиции: focus\_on\_rural\_communities — 5 фр.; focus on rural development — 3 фр.; supports climate adaptation finance — 3 фр.; development\_oriented — 2 фр.; focus on implementation of global climate commitments — 2 фр.; implementation\_oriented — 2 фр.; prioritizes risk reduction and investment attraction for rural sector — 2 фр.; results\_oriented — 2 фр..

### Ildephonse Musafiri

Фрагментов: **4**, допустимых опор позиции: **4**. Публикаций с опорами позиции: 4; доменов: 4. Фрагментов без явной даты события: 4.

Кандидаты по допустимым опорам позиции: gender-responsive policymaking — 2 фр.; agricultural reform design — 1 фр.; attention to women smallholders — 1 фр.; concern for women smallholders — 1 фр.; engagement\_in\_agricultural\_policy — 1 фр.; participation\_in\_international\_forums — 1 фр.; prioritizes nutritional value over taste in seed policy — 1 фр.; pro-technology in agriculture — 1 фр..

### Jean Claude Musabyimana

Фрагментов: **2**, допустимых опор позиции: **1**. Публикаций с опорами позиции: 1; доменов: 1. Фрагментов без явной даты события: 2.

Кандидаты по допустимым опорам позиции: emphasis on legal compliance — 1 фр.; procedural focus — 1 фр.; support for institutionalised electoral processes — 1 фр..

### Jean-Damascène Bizimana

Фрагментов: **96**, допустимых опор позиции: **70**. Публикаций с опорами позиции: 66; доменов: 23. Фрагментов без явной даты события: 88.

Кандидаты по допустимым опорам позиции: diplomatic\_engagement — 3 фр.; patriotism — 3 фр.; commemorative — 2 фр.; diplomatic — 2 фр.; emphasis on historical legacy — 2 фр.; forward\_planning — 2 фр.; genocide\_prevention — 2 фр.; national unity — 2 фр..

### Clare Akamanzi

Фрагментов: **121**, допустимых опор позиции: **63**. Публикаций с опорами позиции: 49; доменов: 34. Фрагментов без явной даты события: 121.

Кандидаты по допустимым опорам позиции: supportive\_of\_entrepreneurship — 16 фр.; supportive of entrepreneurship — 8 фр.; pro\_innovation — 7 фр.; supportive\_of\_innovation — 7 фр.; pro-innovation — 5 фр.; supportive\_of\_creative\_industries — 4 фр.; optimistic\_about\_innovation — 3 фр.; pan\_african\_outlook — 3 фр..

### Francis Gatare

Фрагментов: **37**, допустимых опор позиции: **28**. Публикаций с опорами позиции: 23; доменов: 16. Фрагментов без явной даты события: 36.

Кандидаты по допустимым опорам позиции: supports women in political leadership — 7 фр.; advocates professional networking — 2 фр.; emphasizes peer learning and mutual support — 2 фр.; values professional networking — 2 фр.; advocates political training and capacity building — 1 фр.; advocates professional capacity-building — 1 фр.; capacity\_building\_focus — 1 фр.; career-focused — 1 фр..

### Gertrude Kazarwa

Фрагментов: **20**, допустимых опор позиции: **11**. Публикаций с опорами позиции: 11; доменов: 5. Фрагментов без явной даты события: 19.

Кандидаты по допустимым опорам позиции: development-focused — 2 фр.; diplomatic — 1 фр.; acknowledges remaining gender gaps — 1 фр.; backs\_security\_for\_development — 1 фр.; commemorative — 1 фр.; concern for basic services — 1 фр.; concern for public services — 1 фр.; coopérative — 1 фр..

### Donatille Mukabalisa

Фрагментов: **3**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 3.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### François-Xavier Kalinda

Фрагментов: **13**, допустимых опор позиции: **9**. Публикаций с опорами позиции: 8; доменов: 3. Фрагментов без явной даты события: 12.

Кандидаты по допустимым опорам позиции: promotion de la mémoire du génocide — 2 фр.; anti\_discrimination — 1 фр.; appel au calme pendant les commémorations — 1 фр.; commemorative — 1 фр.; critical\_of\_past\_regimes — 1 фр.; emphasizes legacy of service — 1 фр.; engagement in inter-parliamentary diplomacy — 1 фр.; engagement in interparliamentary diplomacy — 1 фр..

### Augustin Iyamuremye

Фрагментов: **1**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 1.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Alvera Mukabaramba

Фрагментов: **3**, допустимых опор позиции: **2**. Публикаций с опорами позиции: 2; доменов: 2. Фрагментов без явной даты события: 3.

Кандидаты по допустимым опорам позиции: community\_building — 1 фр.; cooperation — 1 фр.; regional\_integration — 1 фр.; sports\_diplomacy — 1 фр.; supports promotion of professional livestock farming — 1 фр.; supports small livestock production — 1 фр.; supports stricter regulation of veterinary medicines — 1 фр..

### Mussa Fazil Harerimana

Фрагментов: **5**, допустимых опор позиции: **2**. Публикаций с опорами позиции: 2; доменов: 2. Фрагментов без явной даты события: 5.

Кандидаты по допустимым опорам позиции: concern\_for\_public\_service\_delivery — 1 фр.; favorable to religious inclusion in national celebrations — 1 фр.; focus\_on\_digital\_inclusion — 1 фр.; supportive of Muslim religious rights — 1 фр..

### Pudence Rubingisa

Фрагментов: **15**, допустимых опор позиции: **10**. Публикаций с опорами позиции: 10; доменов: 9. Фрагментов без явной даты события: 15.

Кандидаты по допустимым опорам позиции: concern\_for\_victims — 2 фр.; support for youth entrepreneurship — 2 фр.; communicative — 1 фр.; concern\_for\_public\_safety — 1 фр.; cooperation\_with\_china — 1 фр.; emergency\_responsiveness — 1 фр.; emphasis on knowledge sharing among youth — 1 фр.; engagement\_with\_liberation\_memory — 1 фр..

### Samuel Dusengiyumva

Фрагментов: **44**, допустимых опор позиции: **25**. Публикаций с опорами позиции: 23; доменов: 11. Фрагментов без явной даты события: 40.

Кандидаты по допустимым опорам позиции: public\_engagement — 2 фр.; administrative — 1 фр.; city\_diplomacy — 1 фр.; civic\_engagement — 1 фр.; commemoration\_support — 1 фр.; commemorative — 1 фр.; community\_involvement — 1 фр.; concerned about illicit alcohol — 1 фр..

### Frank Habineza

Фрагментов: **35**, допустимых опор позиции: **25**. Публикаций с опорами позиции: 18; доменов: 10. Фрагментов без явной даты события: 34.

Кандидаты по допустимым опорам позиции: institutional opposition — 2 фр.; pro-democracy — 2 фр.; accepts institutional responsibility — 1 фр.; advocacy for citizens — 1 фр.; advocates-continued-green-policies — 1 фр.; ambitious — 1 фр.; anti\_divisive\_politics — 1 фр.; commitment to public service — 1 фр..

### Philippe Mpayimana

Фрагментов: **0**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 0.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Victoire Ingabire Umuhoza

Фрагментов: **92**, допустимых опор позиции: **45**. Публикаций с опорами позиции: 36; доменов: 26. Фрагментов без явной даты события: 86.

Кандидаты по допустимым опорам позиции: opposition\_leader — 4 фр.; litigious — 4 фр.; civil\_liberties — 2 фр.; civil\_liberties\_advocate — 2 фр.; défense de l'État de droit — 2 фр.; legal\_challenge — 2 фр.; opposition leader — 2 фр.; vulnerability — 2 фр..

### Diane Shima Rwigara

Фрагментов: **9**, допустимых опор позиции: **0**. Публикаций с опорами позиции: 0; доменов: 0. Фрагментов без явной даты события: 9.

Допустимых опор собственной публичной позиции пока нет. Это неизвестность в данном корпусе, а не нулевая выраженность свойств.

### Bernard Ntaganda

Фрагментов: **4**, допустимых опор позиции: **3**. Публикаций с опорами позиции: 1; доменов: 1. Фрагментов без явной даты события: 4.

Кандидаты по допустимым опорам позиции: critical of international organisation — 2 фр.; concerned about linguistic policy — 1 фр.; critical of human rights situation — 1 фр.; oppositional stance — 1 фр.; pro-democracy — 1 фр..

### Christopher Kayumba

Фрагментов: **1**, допустимых опор позиции: **1**. Публикаций с опорами позиции: 1; доменов: 1. Фрагментов без явной даты события: 1.

Кандидаты по допустимым опорам позиции: critical\_of\_sanctions\_against\_RDF — 1 фр.; positions\_on\_international\_awards — 1 фр.; supportive\_of\_RDF — 1 фр..

### Paul Rusesabagina

Фрагментов: **58**, допустимых опор позиции: **21**. Публикаций с опорами позиции: 21; доменов: 15. Фрагментов без явной даты события: 58.

Кандидаты по допустимым опорам позиции: protective — 7 фр.; courageous — 6 фр.; humanitarian — 3 фр.; resourceful — 3 фр.; protector — 1 фр.; rescuer — 1 фр.; advocacy-related travel — 1 фр.; altruistic — 1 фр..

### Faustin Kayumba Nyamwasa

Фрагментов: **7**, допустимых опор позиции: **3**. Публикаций с опорами позиции: 3; доменов: 3. Фрагментов без явной даты события: 7.

Кандидаты по допустимым опорам позиции: alleged\_terrorism\_financier — 1 фр.; exiled opposition figure — 1 фр.; founder — 1 фр.; RNC\_leadership — 1 фр.; sanctioned\_individual — 1 фр.; target of attacks — 1 фр..

### Louise Mushikiwabo

Фрагментов: **347**, допустимых опор позиции: **147**. Публикаций с опорами позиции: 123; доменов: 71. Фрагментов без явной даты события: 344.

Кандидаты по допустимым опорам позиции: ambitious — 15 фр.; incumbent — 5 фр.; seeking\_reelection — 5 фр.; diplomatic — 5 фр.; international\_engagement — 4 фр.; seeks\_continuity — 4 фр.; politically\_ambitious — 3 фр.; ambition — 3 фр..

### Claver Gatete

Фрагментов: **97**, допустимых опор позиции: **72**. Публикаций с опорами позиции: 47; доменов: 32. Фрагментов без явной даты события: 95.

Кандидаты по допустимым опорам позиции: development-oriented — 5 фр.; pro-innovation — 5 фр.; future-oriented — 3 фр.; pro-growth — 3 фр.; pro-industrialization — 3 фр.; pro-technology — 3 фр.; reformist — 3 фр.; strategic — 3 фр..

### Valentine Rugwabiza

Фрагментов: **9**, допустимых опор позиции: **6**. Публикаций с опорами позиции: 3; доменов: 3. Фрагментов без явной даты события: 9.

Кандидаты по допустимым опорам позиции: cautious — 1 фр.; control-oriented — 1 фр.; coopérative avec le gouvernement hôte — 1 фр.; demanding-strict-evidence — 1 фр.; engagée pour la continuité institutionnelle — 1 фр.; favorable à la participation nationale — 1 фр.; incrementalist — 1 фр.; institutional-loyal — 1 фр..

## Границы результата

- Это машинный черновик: найденные характеристики и группы синонимов не утверждены пользователем.
- Число фрагментов, публикаций и источников не является интенсивностью политического свойства.
- Повторы одного события и независимость источников требуют отдельной проверки; домены не равны независимым источникам.
- Отсутствие свидетельств и статусы review/error/uncertain не означают нулевое свойство политика.
- Объединены только буквальные совпадения названий и заявленных aliases; смысловая эквивалентность не установлена.
- Покрытие done относится к подготовленным заданиям; полный отбор и извлечение показаны отдельно.
- Расход рассчитан по API usage и зафиксированным тарифам; окончательную оплату сверяют с биллингом.
- В машинных таблицах названия сохранены на языке модели; русский редакционный список приложен отдельно, утверждение словаря ещё требуется.
