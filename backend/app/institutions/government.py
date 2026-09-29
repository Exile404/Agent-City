"""Who pays for the university: income tax and the training levy.

Income tax comes off every wage except the minimum wage and pays the stipend.
Employers add a training levy on top of every wage they pay, straight into the
university's fund, which with tuition pays the need-based scholarship. When
the fund runs short the government pays the rest, so the treasury shows
whether the taxes cover the city's promises. The simulation owns every rate.
"""

#: Share of every wage taken as income tax. The porter's minimum wage is exempt.
INCOME_TAX = 0.05
#: Share of every wage an employer adds as a training levy for the university.
TRAINING_LEVY = 0.04
