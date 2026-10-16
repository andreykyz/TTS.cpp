#pragma once
#include <string>
#include <utility>
namespace ru_test {
static const std::pair<std::string,std::string> BRACKETS[] = {
    { "Здравствуйте! Как ваши дела?", "Здравствуйте! Как ваши дела?" },
    { "Это было большое окно.", "Это было большое окно." },
    { "Замок открыт, а замок на двери.", "Замок открыт, а замок на двери." },
    { "Я чувствую себя очень счастливо.", "Я чувствую себя очень счастливо." },
    { "Солнце светит ярко сегодня.", "Солнце светит ярко сегодня." },
    { "Вчера было скучно и дорого.", "Вчера было скучно и дорого." },
    { "Много людей пришло на праздник.", "Много людей пришло на праздник." },
    { "Яичница с беконом на завтрак.", "Яичница с беконом на завтрак." },
    { "Конечно, я помогу тебе.", "Конечно, я помогу тебе." },
    { "Щавель и щётка на столе.", "Щавель и щётка на столе." },
    { "Она сказала, что это было интересно.", "Она сказала, что это было интересно." },
    { "Привет [мир] и {земля}!", "Привет (мир) и (земля)!" },
};
static const int BRACKETS_N = 12;
static const std::pair<std::string,std::string> PLUS[] = {
    { "Здр+авствуйте! К+ак в+аши дел+а?", "Здра́вствуйте! Ка́к ва́ши дела́?" },
    { "Это б+ыло больш+ое окн+о.", "Это бы́ло большо́е окно́." },
    { "З+амок откр+ыт, а зам+ок н+а двер+и.", "За́мок откры́т, а замо́к на́ двери́." },
    { "Я ч+увствую себ+я +очень сч+астливо.", "Я чу́вствую себя́ о́чень сча́стливо." },
    { "С+олнце светит +ярко сег+одня.", "Со́лнце светит я́рко сего́дня." },
    { "Вчер+а было ск+учно и д+орого.", "Вчера́ было ску́чно и до́рого." },
    { "Мн+ого люд+ей пришл+о на пр+аздник.", "Мно́го люде́й пришло́ на пра́здник." },
    { "Я+ичница с бек+оном на з+автрак.", "Яи́чница с беко́ном на за́втрак." },
    { "Кон+ечно, я помог+у теб+е.", "Коне́чно, я помогу́ тебе́." },
    { "Щав+ель и щётка на стол+е.", "Щаве́ль и щётка на столе́." },
    { "Он+а сказ+ала, что +это б+ыло интер+есно.", "Она́ сказа́ла, что э́то бы́ло интере́сно." },
    { "Прив+ет(м+ир) и(земл+я)!", "Приве́т(ми́р) и(земля́)!" },
};
static const int PLUS_N = 12;
static const std::pair<std::string,std::string> RESPELL[] = {
    { "здра́вствуйте! ка́к ва́ши дела́?", "здра́ствуйте! ка́к ва́ши дела́?" },
    { "это бы́ло большо́е окно́.", "это бы́ло большо́е окно́." },
    { "за́мок откры́т, а замо́к на́ двери́.", "за́мок откры́т, а замо́к на́ двери́." },
    { "я чу́вствую себя́ о́чень сча́стливо.", "я чу́ствую себя́ о́чень сча́сливо." },
    { "со́лнце светит я́рко сего́дня.", "со́нце светит я́рко сего́дня." },
    { "вчера́ было ску́чно и до́рого.", "вчера́ было ску́шно и до́рога." },
    { "мно́го люде́й пришло́ на пра́здник.", "мно́го люде́й пришло́ на пра́здник." },
    { "яи́чница с беко́ном на за́втрак.", "яи́шница с беко́ном на за́втрак." },
    { "коне́чно, я помогу́ тебе́.", "коне́шно, я помогу́ тебе́." },
    { "щаве́ль и щётка на столе́.", "щаве́ль и щётка на столе́." },
    { "она́ сказа́ла, что э́то бы́ло интере́сно.", "она́ сказа́ла, что э́то бы́ло интере́сно." },
    { "приве́т(ми́р) и(земля́)!", "приве́т(ми́р) и(земля́)!" },
};
static const int RESPELL_N = 12;
static const std::pair<std::string,std::string> NORMALIZE[] = {
    { "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲiɭˈɑ?", "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲilˈɑ?" },
    { "ˈɛtʌ bˈyɭʌ bʌɭʃˈoji ʌknˈo.", "ˈɛtʌ bˈylʌ bʌlʃˈoji ʌknˈo." },
    { "zˈɑmʌk ʌtkrˈyt, a zamˈok nˈɑ dvʲirʲˈi.", "zˈɑmʌk ʌtkrˈyt, a zamˈok nˈɑ dvʲirʲˈi." },
    { "ˈja ʧʲˈu\"stvuju sʲibʲˈɑ ˈoʧʲin ɕˈɑsɭʲivʌ.", "ˈja ʧˈustvuju sʲibʲˈɑ ˈoʧin ɕˈɑslʲivʌ." },
    { "sˈonʦy svʲitʲˈit ˈjarkʌ sʲivˈodnʲʌ.", "sˈonʦy svʲitʲˈit ˈjarkʌ sʲivˈodnʲʌ." },
    { "fʧʲirˈɑ bˌyɭʌ skˈuʃnʌ ɪ dˈorʌɡa.", "fʧirˈɑ bˌylʌ skˈuʃnʌ ɪ dˈorʌɡa." },
    { "mnˈoɡʌ ɭʲu\"dʲˈej prʲiʃɭˈo nə prˈɑznʲik.", "mnˈoɡʌ lʲudʲˈej prʲiʃlˈo nə prˈɑznʲik." },
    { "jaˈiʃnʲiʦa s bʲikˈonʌm nə zˈɑftrʌk.", "jaˈiʃnʲiʦa s bʲikˈonʌm nə zˈɑftrʌk." },
    { "kʌnʲˈeʃnʌ, ˈja pʌmʌɡˈu tʲibʲˈe.", "kʌnʲˈeʃnʌ, ˈja pʌmʌɡˈu tʲibʲˈe." },
    { "ɕavʲˈeɭʲ ɪ ɕˈɵtka nə stʌɭʲˈe.", "ɕavʲˈelʲ ɪ ɕˈotka nə stʌlʲˈe." },
    { "ʌnˈɑ skazˈɑɭa, ʃto ˈɛtʌ bˈyɭʌ ɪntʲirʲˈesnʌ.", "ʌnˈɑ skazˈɑla, ʃto ˈɛtʌ bˈylʌ ɪntʲirʲˈesnʌ." },
    { "prʲivʲˈet(mʲˈir) ˈi(ʑimɭʲˈɑ)!", "prʲivʲˈet(mʲˈir) ˈi(ʒimlʲˈɑ)!" },
};
static const int NORMALIZE_N = 12;
static const std::pair<std::string,std::string> LENGTHEN[] = {
    { "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲilˈɑ?", "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲilˈɑ?" },
    { "ˈɛtʌ bˈylʌ bʌlʃˈoji ʌknˈo.", "ˈɛtʌ bˈylʌ bʌlʃˈoji ʌknˈo." },
    { "zˈɑmʌk ʌtkrˈyt, a zamˈok nˈɑ dvʲirʲˈi.", "zˈɑmʌk ʌtkrˈyt, a zamˈok nˈɑ dvʲirʲˈi." },
    { "ˈja ʧˈustvuju sʲibʲˈɑ ˈoʧin ɕˈɑslʲivʌ.", "ˈja ʧˈustvuju sʲibʲˈɑ ˈoʧin ɕːˈɑslʲivʌ." },
    { "sˈonʦy svʲitʲˈit ˈjarkʌ sʲivˈodnʲʌ.", "sˈonʦy svʲitʲˈit ˈjarkʌ sʲivˈodnʲʌ." },
    { "fʧirˈɑ bˌylʌ skˈuʃnʌ ɪ dˈorʌɡa.", "fʧirˈɑ bˌylʌ skˈuʃnʌ ɪ dˈorʌɡa." },
    { "mnˈoɡʌ lʲudʲˈej prʲiʃlˈo nə prˈɑznʲik.", "mnˈoɡʌ lʲudʲˈej prʲiʃlˈo nə prˈɑznʲik." },
    { "jaˈiʃnʲiʦa s bʲikˈonʌm nə zˈɑftrʌk.", "jaˈiʃnʲiʦa s bʲikˈonʌm nə zˈɑftrʌk." },
    { "kʌnʲˈeʃnʌ, ˈja pʌmʌɡˈu tʲibʲˈe.", "kʌnʲˈeʃnʌ, ˈja pʌmʌɡˈu tʲibʲˈe." },
    { "ɕavʲˈelʲ ɪ ɕˈotka nə stʌlʲˈe.", "ɕːavʲˈelʲ ɪ ɕːˈotka nə stʌlʲˈe." },
    { "ʌnˈɑ skazˈɑla, ʃto ˈɛtʌ bˈylʌ ɪntʲirʲˈesnʌ.", "ʌnˈɑ skazˈɑla, ʃto ˈɛtʌ bˈylʌ ɪntʲirʲˈesnʌ." },
    { "prʲivʲˈet(mʲˈir) ˈi(ʒimlʲˈɑ)!", "prʲivʲˈet(mʲˈir) ˈi(ʒimlʲˈɑ)!" },
};
static const int LENGTHEN_N = 12;
static const std::pair<std::string,std::string> REDUCE[] = {
    { "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲilˈɑ?", "zdrˈɑstvujtʲi! kˈɑk vˈɑʃy dʲilˈɑ?" },
    { "ˈɛtʌ bˈylʌ bʌlʃˈoji ʌknˈo.", "ˈɛtə bˈylə bɐlʃˈoji ɐknˈo." },
    { "zˈɑmʌk ʌtkrˈyt, a zamˈok nˈɑ dvʲirʲˈi.", "zˈɑmək ɐtkrˈyt, ɐ zɐmˈok nˈɑ dvʲirʲˈi." },
    { "ˈja ʧˈustvuju sʲibʲˈɑ ˈoʧin ɕːˈɑslʲivʌ.", "ˈja ʧˈustvuju sʲibʲˈɑ ˈoʧin ɕːˈɑslʲivə." },
    { "sˈonʦy svʲitʲˈit ˈjarkʌ sʲivˈodnʲʌ.", "sˈonʦy svʲitʲˈit ˈjarkə sʲivˈodnʲɪ." },
    { "fʧirˈɑ bˌylʌ skˈuʃnʌ ɪ dˈorʌɡa.", "fʧirˈɑ bˌylə skˈuʃnə ɪ dˈorəɡə." },
    { "mnˈoɡʌ lʲudʲˈej prʲiʃlˈo nə prˈɑznʲik.", "mnˈoɡə lʲudʲˈej prʲiʃlˈo nə prˈɑznʲik." },
    { "jaˈiʃnʲiʦa s bʲikˈonʌm nə zˈɑftrʌk.", "jɪˈiʃnʲiʦə s bʲikˈonəm nə zˈɑftrək." },
    { "kʌnʲˈeʃnʌ, ˈja pʌmʌɡˈu tʲibʲˈe.", "kɐnʲˈeʃnə, ˈja pəmɐɡˈu tʲibʲˈe." },
    { "ɕːavʲˈelʲ ɪ ɕːˈotka nə stʌlʲˈe.", "ɕːɐvʲˈelʲ ɪ ɕːˈotkə nə stɐlʲˈe." },
    { "ʌnˈɑ skazˈɑla, ʃto ˈɛtʌ bˈylʌ ɪntʲirʲˈesnʌ.", "ɐnˈɑ skɐzˈɑlə, ʃtə ˈɛtə bˈylə ɪntʲirʲˈesnə." },
    { "prʲivʲˈet(mʲˈir) ˈi(ʒimlʲˈɑ)!", "prʲivʲˈet(mʲˈir) ˈi(ʒimlʲˈɑ)!" },
};
static const int REDUCE_N = 12;
}  // namespace ru_test
