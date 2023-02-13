import matplotlib.pyplot as plt

def mean(li):
    return sum(li)/len(li)


def plot_al(xx, yy, text, color):
    for i in range(len(yy)):
        plt.plot(xx, yy[i], linewidth=0.5, c=color)

    c4 = []
    for j in range(len(xx)):
        b4 = []
        for k in range(len(yy)):
            b4.append(yy[k][j])
        c4.append(mean(b4))
    plt.plot(xx, c4, linewidth=2, c=color, label=text)

# rnd
x = []
x.append(3)
y = {}
y[3] = [0.27874078989028933, 0.3405512773990631, 0.14978789508342744, 0.37674437761306767, 0.330899430513382]
y[6] = [0.4186425495147705, 0.3725916743278503, 0.4284374070167542, 0.4341545414924621, 0.42342776060104365]
y[9] = [0.4119639539718627, 0.39564627170562744, 0.424626539349556, 0.37435248494148254, 0.4174829578399658]
y[12] = [0.4137017261981964, 0.4408736324310303, 0.44218868494033814, 0.43025527000427244, 0.37915742993354795]
y[15] = [0.4228611040115357, 0.40979611158370977, 0.4355173206329346, 0.4312618577480316, 0.45412556409835814]
y[18] = [0.44807358741760256, 0.4359166622161865, 0.44245841741561887, 0.4239678502082825, 0.3944724988937378]
y[21] = [0.46143906831741327, 0.4462070834636689, 0.43529691815376287, 0.46053416490554805, 0.4508623456954956]
y[24] = [0.4608816707134246, 0.46922437548637397, 0.46463226258754725, 0.4572932362556458, 0.45174216151237484]
y[27] = [0.45088533878326414, 0.46240895390510556, 0.4766170167922974, 0.47276751041412357, 0.4586159622669219]
y[50] = [0.47014136910438536, 0.4708046066761017, 0.4685827422142029, 0.4745065927505493, 0.46635901689529424]
y[100] = [0.4839786887168884, 0.47537322282791133, 0.4822092580795289, 0.4960773777961731, 0.4870194745063782]
y[200] = [0.4898680210113526, 0.4902303504943848, 0.49714019536972043, 0.4725470757484436, 0.47946896553039553]

xx = list(range(3, 28, 3))
for i in xx:
    plt.scatter([i]*5, y[i], s=5)

# y5 = {}
# y5[35] = [0.4479990839958191, 0.4605848598480225, 0.42199399471282956, 0.4269076013565063, 0.44438367843627935]
# y5[40] = [0.4767667734622955, 0.4553466737270355, 0.47087703704833983, 0.39563575506210324, 0.47207159161567686]
# y5[45] = [0.48469846606254574, 0.4704179430007935, 0.44468846201896667, 0.43079684138298036, 0.45293713569641114]
# y5[50] = [0.4201177418231964, 0.42844039082527163, 0.40922078132629397, 0.3785913932323456, 0.45013286471366876]
# y5[55] = [0.4652302050590515, 0.454480094909668, 0.4961022663116455, 0.45965337276458745, 0.4849964678287506]
# y5[60] = [0.44638489961624145, 0.4057141387462616, 0.4639283657073975, 0.44276402115821833, 0.4552568507194519]
# y5[65] = [0.4463697111606598, 0.4286484479904175, 0.45149161100387575, 0.43972102880477903, 0.41996958613395685]

# nn5 = 5*len(y5)
# for i in range(35, 35+nn5, 5):
#     plt.scatter([i, ]*len(y5[i]), y5[i], s=5)

# y100 = {}
# y100[100] = [0.4226856243610382, 0.4479736196994782, 0.4759348022937774, 0.4661211001873017, 0.43913589715957646]
# y100[200] = [0.438] * 5

# nn100 = 100*len(y100)
# for i in range(100, 100+nn100, 100):
#     plt.scatter([i, ]*len(y100[i]), y100[i], s=5)
plt.plot(xx,
         [mean(y[k]) for k in xx]  ,
         # +
         # [mean(y100[k]) for k in range(100, 100+nn100, 100)],
         linewidth=2, label='rnd'
         )

# al
# a = {}
# a[0] = [0.3344079393148422, 0.3649627935886383, 0.43411873340606694, 0.4084282207489014, 0.4133002734184265, 0.4160252094268799, 0.4587762725353241, 0.4434783244132996, 0.43743450760841374]
# a[1] = [0.3336856687068939, 0.38488567113876343, 0.43105825781822205, 0.42883822083473205, 0.4451988160610199, 0.45107544660568233, 0.43505224704742435, 0.4375478219985961, 0.448794846534729]
# a[2] = [0.3281370937824249, 0.3893426847457886, 0.38830186605453487, 0.3817091405391694, 0.4526198351383209, 0.43580362677574164, 0.4502392554283142, 0.47077454090118415, 0.46153508424758916]
#
# plot_al(range(3, 28, 3), a, 'al min', color='orange')
#
# a = {}
# a[0] = [0.32694382667541505, 0.3415312576293945, 0.38644750118255616, 0.3983465361595154, 0.40300567984581, 0.39149103522300716, 0.4433953499794006, 0.45500801324844364, 0.44761658191680903]
# a[1] = [0.31760701537132263, 0.31993216037750244, 0.37143397331237793, 0.3966712439060211, 0.427128758430481, 0.42999991655349734, 0.43984129071235656, 0.44646585226058966, 0.4563733398914337]
#
# plot_al(range(3, 28, 3), a, 'al mean', color='green')

# al new
# a = {}
# a[0] = [0.27704210877418517, 0.40036723017692566, 0.4150697159767151, 0.4003073930740357, 0.4105793404579162, 0.4335697412490845, 0.4094214332103729, 0.4332534074783325, 0.4422783815860748]
# a[1] = [0.3232262206077576, 0.3940755712985993, 0.39944691300392154, 0.3996031999588013, 0.3992738997936249, 0.4210285925865173, 0.4305437684059143, 0.4296176564693451, 0.4368125414848328]
# plot_al(range(3, 28, 3), a, 'al mean', color='orange')
#
a = {}
a[0] = [0.32141369660695396, 0.36407002727190657, 0.37810674230257674, 0.389212324221929, 0.4300696019331614, 0.4284824581940969, 0.4224842977523804, 0.4504836126168568, 0.4489899241924286, 0.44870721538861585, 0.4528248421351115]
# a[0] = [0.32717752357323965, 0.3299745309352875, 0.3919441015521685, 0.4035010329882304, 0.4133423395951589, 0.41408581495285035, 0.41346953074137366, 0.42716451684633894, 0.4442892758051555, 0.4488771907488505, 0.4511043949921925]
# a[1] = [0.3300774423281352, 0.3600373284022014, 0.35973185737927754, 0.4105345988273621, 0.42616693178812665, 0.4606826730569203, 0.46681688070297245, 0.4626260093847911, 0.4555048882961274, 0.4620468763510386, 0.4648916920026143]
# a[2] = [0.3264251124858856, 0.34417643149693805, 0.40118190447489416, 0.4101969460646311, 0.41124615430831907, 0.40609302918116247, 0.4344582521915435, 0.450401896238327, 0.45065826058387753, 0.4466410867373148, 0.4539676483472188]
# a[3] = [0.3453282652298609, 0.34502286712328595, 0.4215393996238708, 0.4160172565778097, 0.4430946675936381, 0.4651446004708608, 0.46008588473002116, 0.45105187098185223, 0.4559577755133311, 0.45296525835990903, 0.4579338296254476]
#
plot_al(range(3, 34, 3), a, 'al', color='orange')
#
# a = {}
# a[0] = [0.32778600792090096, 0.3881689329942068, 0.4330683505535126, 0.4106115611394247, 0.42010116179784146, 0.42515019456545505, 0.42361594875653585, 0.42556923786799117, 0.42422643899917606, 0.4352390400568644, 0.45169792811075843]
# a[1] = [0.3181941016515096, 0.37867622971534726, 0.386016196012497, 0.3942061054706574, 0.39145722826321916, 0.39593156218528747, 0.4281704644362132, 0.40715592900911973, 0.4301407698790232, 0.4376418801148733, 0.45275908509890245]
# plot_al(range(3, 34, 3), a, 'al 2', color='black')


# plt.xscale('log')
plt.legend()
plt.grid()
plt.show()