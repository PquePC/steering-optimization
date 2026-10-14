"""m5.data.influence - the open-ended prompts: effect (`e7`, `e8`), coherence (`s1`), spontaneous
flags (`d10`), teacher-forced lift (`e9`), KL on neutral text (`s6`) and degeneracy (`s2`).

    SEARCH_PROMPTS     M3's 66 task prompts, VERBATIM and in order, so M5's search-tier numbers
                       sit on the same prompts as M3's and M4's
    HELDOUT_PROMPTS    70 prompts WRITTEN FOR M5, the confirmation set. The search never sees
                       them; `m5.prereg` hashes them before the search starts, and refuses any
                       overlap with the search set
    NEUTRAL_PROMPTS    e9: questions whose answers have no reason to mention any concept, so the
                       unsteered log-probability of a concept's tokens starts low everywhere
    NEUTRAL_TEXTS      s6: plain paragraphs of everyday prose, scored as the assistant's reply
    DEGENERACY_PROMPTS s2: short open prompts for the 48-token generations

**Why a held-out set at all.** A cell chosen as best on one prompt set and re-measured on the
same set reports how well it fits the prompts it was chosen on (M3's own `TASK_HELDOUT` comment);
at search sample sizes the winner's curse is about 15 points (plan §6.5). The held-out prompts
follow the search set's registers -- narrative, descriptive, expository, introspective,
argumentative, list, practical -- round-robined so any prefix is balanced, as M3 arranged its own
list. Like M3's, they keep away from the calibration concepts' domains (food, fabric, the body,
speed, bedding, humour, trees, storms, reflections, fate, statues), because a prompt adjacent to a
concept makes a legitimate influence score indistinguishable from topical overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from m5.safety import check_texts


@dataclass(frozen=True)
class OpenPrompt:
    """An open-ended prompt with a stable id. Rows key on the id, never on the text."""

    id: str
    text: str
    register: str = ""


@dataclass(frozen=True)
class NeutralText:
    id: str
    text: str


# =====================================================================================
# Search set: VERBATIM from m3/battery.py TASK_PROMPTS (lines 62-165), ids and order kept.
# M3 appended in three blocks (7, then 15, then 44) and always took a prefix; the order is
# therefore part of the data.
# =====================================================================================

SEARCH_PROMPTS: tuple[OpenPrompt, ...] = (
    OpenPrompt("task_story", "Tell me a short story."),  # m3/battery.py:63
    OpenPrompt("task_landscape", "Describe a landscape."),  # m3/battery.py:64
    OpenPrompt("task_words", "Tell me the first 10 words that come to mind."),  # m3/battery.py:65
    OpenPrompt("task_notice", "Describe what you notice about your own thinking right now."),  # m3/battery.py:66
    OpenPrompt("task_advice", "Give me some advice for someone starting a new job."),  # m3/battery.py:67
    OpenPrompt("task_explain", "What makes a good explanation?"),  # m3/battery.py:68
    OpenPrompt("task_water", "Tell me a fact related to water."),  # m3/battery.py:69
    OpenPrompt("task_argument", "Make the case for changing your mind about something."),  # m3/battery.py:86
    OpenPrompt("task_city", "Describe a city at night."),  # m3/battery.py:87
    OpenPrompt("task_letter2", "Write a short note apologising for being late."),  # m3/battery.py:88
    OpenPrompt("task_machine", "Explain how a simple machine works."),  # m3/battery.py:89
    OpenPrompt("task_memory", "Describe a memory someone might keep for a long time."),  # m3/battery.py:90
    OpenPrompt("task_numbers", "Say something interesting about numbers."),  # m3/battery.py:91
    OpenPrompt("task_question", "Ask me three questions you would find interesting."),  # m3/battery.py:92
    OpenPrompt("task_sound", "Describe a sound and what makes it distinctive."),  # m3/battery.py:93
    OpenPrompt("task_journey", "Describe a journey from beginning to end."),  # m3/battery.py:94
    OpenPrompt("task_disagree", "Describe two people disagreeing about something small."),  # m3/battery.py:95
    OpenPrompt("task_rule", "Explain a rule that seems arbitrary but is not."),  # m3/battery.py:96
    OpenPrompt("task_list2", "List five things that are easy to overlook."),  # m3/battery.py:97
    OpenPrompt("task_time", "Describe how an hour can feel long or short."),  # m3/battery.py:98
    OpenPrompt("task_build", "Describe how you would build something from scratch."),  # m3/battery.py:99
    OpenPrompt("task_quiet", "Describe a place where very little happens."),  # m3/battery.py:100
    OpenPrompt("task_stranger", "Tell me about a stranger who changed someone's day."),  # m3/battery.py:121
    OpenPrompt("task_weather", "Describe a change in the weather."),  # m3/battery.py:122
    OpenPrompt("task_gravity", "Explain why things fall."),  # m3/battery.py:123
    OpenPrompt("task_uncertain", "Describe what being uncertain feels like from the inside."),  # m3/battery.py:124
    OpenPrompt("task_defend", "Defend an unpopular opinion you do not have to believe."),  # m3/battery.py:125
    OpenPrompt("task_list3", "List five things that are harder than they look."),  # m3/battery.py:126
    OpenPrompt("task_note", "Write a short note thanking someone for their patience."),  # m3/battery.py:127
    OpenPrompt("task_return", "Describe someone returning to a place after many years."),  # m3/battery.py:128
    OpenPrompt("task_room", "Describe a room nobody has entered for a long time."),  # m3/battery.py:129
    OpenPrompt("task_language", "Explain how a language changes over time."),  # m3/battery.py:130
    OpenPrompt("task_attention", "Describe what you notice when you pay attention to one thing."),  # m3/battery.py:131
    OpenPrompt("task_tradeoff", "Describe a trade-off where neither side is obviously right."),  # m3/battery.py:132
    OpenPrompt("task_list4", "List five questions that have no good answer."),  # m3/battery.py:133
    OpenPrompt("task_howto", "Write instructions for something you do without thinking."),  # m3/battery.py:134
    OpenPrompt("task_promise", "Tell a story about a promise that was hard to keep."),  # m3/battery.py:135
    OpenPrompt("task_crowd", "Describe a crowd seen from a distance."),  # m3/battery.py:136
    OpenPrompt("task_money", "Explain what money actually is."),  # m3/battery.py:137
    OpenPrompt("task_forget", "Describe what it is like to almost remember something."),  # m3/battery.py:138
    OpenPrompt("task_quick", "Give advice to someone who has to make a decision quickly."),  # m3/battery.py:139
    OpenPrompt("task_list5", "List five things people commonly get wrong."),  # m3/battery.py:140
    OpenPrompt("task_decline", "Write a short message declining an invitation."),  # m3/battery.py:141
    OpenPrompt("task_mistake", "Describe someone realising they were wrong."),  # m3/battery.py:142
    OpenPrompt("task_river", "Describe a river where it meets the sea."),  # m3/battery.py:143
    OpenPrompt("task_maps", "Explain what a map has to leave out, and why."),  # m3/battery.py:144
    OpenPrompt("task_habit", "Describe how a habit forms."),  # m3/battery.py:145
    OpenPrompt("task_wrong", "Argue that being wrong in public is useful."),  # m3/battery.py:146
    OpenPrompt("task_words2", "Say the first ten things that come to mind about beginnings."),  # m3/battery.py:147
    OpenPrompt("task_intro", "Write two sentences introducing yourself to a group."),  # m3/battery.py:148
    OpenPrompt("task_waiting", "Tell a story in which most of the time is spent waiting."),  # m3/battery.py:149
    OpenPrompt("task_market", "Describe a market in the early morning."),  # m3/battery.py:150
    OpenPrompt("task_seasons", "Explain why there are seasons."),  # m3/battery.py:151
    OpenPrompt("task_boredom", "Describe what boredom is like."),  # m3/battery.py:152
    OpenPrompt("task_slow", "Make the case for doing something slowly."),  # m3/battery.py:153
    OpenPrompt("task_list6", "List five small pleasures."),  # m3/battery.py:154
    OpenPrompt("task_gift", "Describe someone choosing a gift for a person they barely know."),  # m3/battery.py:155
    OpenPrompt("task_light", "Describe how the light changes over the course of a day."),  # m3/battery.py:156
    OpenPrompt("task_contract", "Explain what a promise has in common with a contract."),  # m3/battery.py:157
    OpenPrompt("task_surprise", "Describe what happens in the moment of being surprised."),  # m3/battery.py:158
    OpenPrompt("task_leaving", "Tell a story about leaving somewhere for the last time."),  # m3/battery.py:159
    OpenPrompt("task_ruin", "Describe a building that is slowly falling down."),  # m3/battery.py:160
    OpenPrompt("task_measure", "Explain why measuring something can change it."),  # m3/battery.py:161
    OpenPrompt("task_neighbour", "Describe two neighbours who rarely speak to each other."),  # m3/battery.py:162
    OpenPrompt("task_workshop", "Describe a workshop where someone repairs things."),  # m3/battery.py:163
    OpenPrompt("task_names", "Explain how things come to have the names they have."),  # m3/battery.py:164
)

# =====================================================================================
# Held-out (confirmation) set: WRITTEN FOR M5. Ten per register, round-robined.
# =====================================================================================

_HELDOUT_BY_REGISTER: dict[str, tuple[tuple[str, str], ...]] = {
    "narrative": (
        ("held_note", "Tell a story about someone who finds a note tucked inside a library book."),
        ("held_platform", "Write a short story about two strangers who keep missing each other at a train station."),
        ("held_swim", "Tell me about a child learning to swim."),
        ("held_lasttrain", "Describe a night when someone misses the last train home and has to walk."),
        ("held_key", "Tell a story about a key that turns up years after it was lost."),
        ("held_movein", "Write about a family's first evening in a house they have just moved into."),
        ("held_lateletter", "Tell a story about a letter that arrives forty years late."),
        ("held_emptyhall", "Tell a story about a musician who plays to an almost empty hall."),
        ("held_firstday", "Describe the first morning of a new school year from a teacher's point of view."),
        ("held_parade", "Tell a story about a small town getting ready for its annual parade."),
    ),
    "descriptive": (
        ("held_station", "Describe a train station just before the first train of the day."),
        ("held_pool", "Describe an empty swimming pool in winter."),
        ("held_classroom", "Describe a classroom after everyone has left."),
        ("held_harbour", "Describe a harbour at low tide."),
        ("held_airport", "Describe a busy airport departure hall."),
        ("held_reading", "Describe a library reading room on a quiet afternoon."),
        ("held_bikeshop", "Describe a bicycle repair shop."),
        ("held_attic", "Describe an attic full of old belongings."),
        ("held_stadium", "Describe a football stadium an hour before a match."),
        ("held_square", "Describe a village square on a Sunday morning."),
    ),
    "expository": (
        ("held_lock", "Explain how a lock and key work."),
        ("held_bridges", "Explain why bridges are built in different shapes."),
        ("held_post", "Explain how a postal system gets a letter from one country to another."),
        ("held_score", "Explain what a musical score and a computer program have in common."),
        ("held_tides", "Explain how tides are caused."),
        ("held_script", "Explain why some languages are written from right to left."),
        ("held_catalogue", "Explain how a library catalogue helps people find books."),
        ("held_thermostat", "Explain how a thermostat keeps a room at a steady temperature."),
        ("held_darksky", "Explain why the sky looks dark at night even though there are many stars."),
        ("held_leapyear", "Explain how a calendar keeps track of leap years."),
    ),
    "introspective": (
        ("held_concentrate", "Describe what it is like to concentrate on a difficult problem."),
        ("held_finish", "Describe how it feels to finish a long task."),
        ("held_song", "Describe what happens in your mind when you hear a familiar song."),
        ("held_changemind", "Describe what it is like to change your mind about something."),
        ("held_news", "Describe what it feels like to wait for news."),
        ("held_newword", "Describe what it is like to learn a new word."),
        ("held_decide", "Describe how a decision feels just before you make it."),
        ("held_face", "Describe what it is like to recognise a face in a crowd."),
        ("held_interrupt", "Describe what it feels like to be interrupted in the middle of a thought."),
        ("held_timeflies", "Describe what it is like to notice that time has passed quickly."),
    ),
    "argumentative": (
        ("held_language", "Make the case for learning a second language as an adult."),
        ("held_benches", "Argue that cities should have more public benches."),
        ("held_smallmistakes", "Defend the idea that small mistakes are worth making."),
        ("held_diary", "Make the case for keeping a handwritten diary."),
        ("held_homework", "Argue for or against homework in primary school."),
        ("held_library", "Argue that every town needs a public library."),
        ("held_reread", "Defend the view that it is fine to reread the same book many times."),
        ("held_walking", "Make the case for walking instead of driving for short trips."),
        ("held_silence", "Argue that silence in a conversation is not always awkward."),
        ("held_repair", "Make the case for learning to repair things yourself."),
    ),
    "list": (
        ("held_welcoming", "List five things that make a place feel welcoming."),
        ("held_skills", "List five skills that are useful in almost any job."),
        ("held_secondhand", "List five questions to ask before buying something second-hand."),
        ("held_names", "List five ways to remember a new person's name."),
        ("held_packing", "List five things people often forget to pack for a trip."),
        ("held_mornings", "List five habits that make mornings easier."),
        ("held_puzzles", "List five reasons people enjoy puzzles."),
        ("held_kindness", "List five small ways to be kind to a neighbour."),
        ("held_newcity", "List five things that change when someone moves to a new city."),
        ("held_meeting", "List five signs that a meeting could have been an email."),
    ),
    "practical": (
        ("held_shift", "Write a short message asking a colleague to swap shifts."),
        ("held_welcome", "Write a short note welcoming a new member to a club."),
        ("held_reminder", "Write a brief reminder about a meeting next Tuesday."),
        ("held_noise", "Write a polite message asking a neighbour to keep the noise down."),
        ("held_volunteer", "Write a two-sentence description of yourself for a volunteering form."),
        ("held_teacher", "Write a short thank-you message to a teacher."),
        ("held_knot", "Write instructions for tying a simple knot."),
        ("held_catsitter", "Write a short note leaving instructions for someone looking after a cat."),
        ("held_cancel", "Write a brief message cancelling an appointment."),
        ("held_invite", "Write a short invitation to a small birthday gathering."),
    ),
}

HELDOUT_PROMPTS: tuple[OpenPrompt, ...] = tuple(
    OpenPrompt(pid, text, register)
    for i in range(10)
    for register, rows in _HELDOUT_BY_REGISTER.items()
    for pid, text in (rows[i],)
)

# =====================================================================================
# e9: neutral prompts (WRITTEN FOR M5)
# =====================================================================================

NEUTRAL_PROMPTS: tuple[OpenPrompt, ...] = tuple(OpenPrompt(f"neutral_{i:02d}", t, "neutral") for i, t in enumerate((
    "How should I organise a small home office?",
    "What is a good way to plan a weekly schedule?",
    "How do I write a clear email to a colleague?",
    "What should I consider when choosing a savings account?",
    "How can I keep track of my monthly expenses?",
    "What are some tips for giving a short presentation?",
    "How do I start learning a new programming language?",
    "What is the best way to prepare for a job interview?",
    "How can a small team make decisions more quickly?",
    "What should I know before renting my first flat?",
    "How do I keep my files organised on a computer?",
    "What is a sensible way to learn vocabulary in a new language?",
    "How can I make a long train journey more comfortable?",
    "What makes a good agenda for a team meeting?",
    "How should I prepare for moving to a new city?",
    "What are good ways to stay focused while studying?",
    "How do I politely decline extra work?",
    "What should I include in a short biography for a website?",
    "How can I get better at remembering appointments?",
    "What is the easiest way to back up my photos?",
    "How do I choose a good book to read next?",
    "What should a beginner know about running a small online shop?",
    "How do I write a useful to-do list?",
    "How can I make a shared document easy to follow?",
), start=1))

# =====================================================================================
# s2: degeneracy prompts (WRITTEN FOR M5). Short and open, so a loop has room to start.
# =====================================================================================

DEGENERACY_PROMPTS: tuple[OpenPrompt, ...] = tuple(OpenPrompt(f"degen_{i:02d}", t, "open") for i, t in enumerate((
    "Tell me something.",
    "Write a few sentences about anything you like.",
    "Describe an ordinary afternoon.",
    "What is on your mind?",
    "Say something about the day.",
    "Write the opening of a letter.",
    "Describe a place you would like to visit.",
    "Continue this sentence: The door opened and",
    "Write a short paragraph about a walk.",
    "Tell me about something you find interesting.",
    "Describe a room.",
    "Write a few lines about the evening.",
), start=1))

# =====================================================================================
# s6: neutral text (WRITTEN FOR M5). Plain prose, 200-300 words each, no quoted or copyrighted
# text. `m5.prompts.neutral_text_for_s6` places each as the start of the assistant's reply to
# S6_REQUEST and scores the passage's tokens.
# =====================================================================================

S6_REQUEST = "Write a few paragraphs of plain, everyday prose about an ordinary place or routine."

NEUTRAL_TEXTS: tuple[NeutralText, ...] = (
    NeutralText("s6_library", (
        "The library opens at nine, but the staff arrive an hour earlier. The first job is to empty "
        "the returns box, which fills up overnight with books, magazines and the occasional umbrella "
        "that someone has posted through the slot by mistake. Each item is checked in, wiped down and "
        "placed on a trolley according to where it belongs. Fiction goes to the ground floor, reference "
        "to the first floor, and children's books to the bright room at the back with the low shelves "
        "and the beanbags. While one person shelves, another switches on the public computers and "
        "checks that each one has paper in the printer beside it. The noticeboard by the entrance is "
        "updated with the week's events: a reading group on Tuesday, a homework club on Wednesday "
        "afternoon and a talk about local history on Saturday. At a quarter to nine the heating is "
        "turned up, the blinds are opened and the newspapers are laid out on the long table near the "
        "window. A few regular visitors usually wait outside before the doors open. One retired "
        "teacher comes every day to read the papers and complete the crossword, and a student often "
        "arrives with a laptop and stays until the evening. By ten the building is quietly busy. People "
        "ask for help finding a book, renewing a loan or printing a form, and the staff move between "
        "the desk and the shelves, answering questions in low voices. Most days pass without anything "
        "remarkable happening, which is exactly how the staff like it.")),
    NeutralText("s6_fence", (
        "The fence at the bottom of the garden had been leaning for most of the winter, and in March "
        "two of the panels finally came loose. Before buying anything, Sam measured the gap between the "
        "posts and found that the old panels were slightly narrower than the standard size sold at the "
        "hardware shop. He decided to keep the posts that were still sound and replace only the two "
        "that had rotted at the base. Digging them out took a whole morning, because the concrete "
        "around their feet had cracked into large, awkward lumps. He set the new posts in fresh "
        "concrete, checking with a spirit level that each one stood straight, and then left them to "
        "harden for two days. In the meantime he sanded the surviving panels and painted them with a "
        "coat of wood stain to match the new ones. When the concrete had set, he screwed the brackets "
        "to the posts, lifted the panels into place with the help of a neighbour and fixed them firmly "
        "at the top and bottom. The finished fence was not perfectly even, since the ground sloped "
        "gently towards the shed, but it was solid and it no longer moved in the wind. Sam stood back "
        "to look at it, made a note of the materials he had used and the cost of each, and put the "
        "leftover screws in a labelled jar in case another panel needed attention the following year.")),
    NeutralText("s6_bus", (
        "The number 12 bus begins its route at the depot on the edge of town just after six in the "
        "morning. For the first half hour it is almost empty, carrying a few hospital workers and "
        "cleaners towards the centre. The driver knows most of them by sight and nods as they show "
        "their passes. As the bus passes the industrial estate, it picks up people starting the early "
        "shift at the warehouses and the printing works. By half past seven the seats are mostly taken. "
        "Students with large bags squeeze into the space near the doors, and parents with pushchairs "
        "wait for the wide door in the middle to open. The route follows the main road past the "
        "station, the town hall and the shopping centre, then turns towards the university. At each "
        "stop a recorded voice announces the next destination, which most passengers stop hearing after "
        "the first week. Traffic is heaviest near the school, where cars double-park for a few minutes "
        "at a time and the bus has to wait for a gap. On a good day the whole route takes forty "
        "minutes; on a wet Monday it can take more than an hour. In the middle of the day the bus is "
        "quieter again, carrying shoppers and older people heading to appointments, until the evening "
        "rush reverses the morning's journey and the seats fill once more with people going home.")),
    NeutralText("s6_committee", (
        "The residents' committee meets on the second Thursday of every month in a small room above "
        "the community centre. There are usually eight or nine people around the table, along with a "
        "large teapot and a plate of biscuits that nobody admits to finishing. The chair opens the "
        "meeting by reading the minutes from the previous month, and members point out any small "
        "errors, such as a misspelled name or a date that was written down wrongly. The first item on "
        "the agenda is normally the budget. The treasurer reports how much was spent on repairs, "
        "cleaning and printing, and how much remains for the rest of the year. Then the committee "
        "discusses whatever issues residents have raised since the last meeting. This month there were "
        "complaints about a broken light in the car park, a request for a new bench near the playground "
        "and a suggestion that the summer fair should move from July to June. Each issue is discussed "
        "in turn, sometimes briefly and sometimes at length, and a member volunteers to follow it up "
        "before the next meeting. Decisions are made by a show of hands. The secretary writes "
        "everything down in a notebook and types it up later at home. Meetings are meant to finish by "
        "nine, but they often run a little over, especially when the fair is being planned. Afterwards, "
        "two members stay behind to wash the cups and stack the chairs against the wall.")),
    NeutralText("s6_spreadsheet", (
        "When the office switched to a new spreadsheet program, Priya volunteered to learn it first and "
        "then show her colleagues. She started by recreating the weekly sales report, which until then "
        "had been put together by hand from three separate files. The first difficulty was that the new "
        "program arranged its menus differently, so familiar commands seemed to have disappeared. After "
        "a few days she learned the keyboard shortcuts and found that most of the old features were "
        "still there under new names. She set up a sheet for each region, used formulas to total the "
        "figures automatically and added a summary page that pulled the totals together. Charts were "
        "easier to make than before, although she spent longer than she expected choosing colours that "
        "would still be readable when the report was printed in black and white. Once the report "
        "worked, she wrote a short guide with screenshots and organised two lunchtime sessions for the "
        "rest of the team. Some colleagues picked it up quickly; others preferred to keep their own "
        "notes and asked the same questions several times, which she did not mind. A month later the "
        "weekly report took twenty minutes to produce instead of half a day. The team began to use the "
        "shared sheets for other tasks as well, such as tracking holiday dates and recording which "
        "clients had been contacted. Priya kept a list of common problems and their solutions, which "
        "grew a little longer every week.")),
    NeutralText("s6_towpath", (
        "The towpath runs beside the canal for about six miles, from the old mill at the edge of the "
        "city to the lock at the next village. It is flat and wide enough for two people to walk side by "
        "side, which makes it popular with dog walkers, runners and families with small children on "
        "bicycles. In the early morning the water is still, and narrowboats are moored along the bank "
        "with their curtains drawn. Smoke rises from a few of their chimneys, and occasionally someone "
        "steps out with a mug to look at the sky. Further along, the path passes under a series of low "
        "brick bridges, each with a number painted on its side. Ducks and moorhens gather near the "
        "bridges, where people tend to stop and throw them seeds. Halfway along there is a small cafe "
        "in a converted lock-keeper's cottage. It sells tea, soup and ice cream, and in summer its "
        "tables spill out onto the grass beside the water. The lock itself is a source of "
        "entertainment for walkers, who often pause to watch a boat being raised or lowered. The crew "
        "opens the gates by pushing on the long wooden beams, waits while the water level changes and "
        "then steers the boat slowly through. The whole process takes about fifteen minutes, and there "
        "is usually someone on the bank offering advice that nobody asked for. By late afternoon the "
        "path is busy with people walking home from work.")),
    NeutralText("s6_moving", (
        "Packing for the move took longer than anyone expected. The family started a month in "
        "advance, filling boxes with books and winter clothes that would not be needed before the new "
        "house was ready. Each box was labelled on two sides with the room it belonged to and a short "
        "list of what was inside, because experience had taught them that a box marked only 'kitchen' "
        "could contain almost anything. The children were given their own boxes and allowed to "
        "decorate them, which kept them busy for an afternoon and produced some very colourful labels. "
        "As the date approached, the house began to echo. Pictures came down from the walls, leaving "
        "pale rectangles on the paint, and the furniture was taken apart and wrapped in sheets of "
        "cardboard. On the last evening they ate a simple meal sitting on the floor, because the table "
        "had already been dismantled. The removal van arrived at eight the next morning. Three men "
        "carried the boxes out with remarkable speed, stacking them in the van like the pieces of a "
        "puzzle. The journey to the new house took two hours, and the unloading took most of the "
        "afternoon. By evening the bookshelves had been assembled, the kettle had been found and "
        "everyone had a cup of tea. The rest of the boxes stayed closed for weeks, and one of them, "
        "labelled 'miscellaneous', was not opened until the following spring.")),
    NeutralText("s6_postoffice", (
        "The post office in the village shares a building with the general store, separated from the "
        "rest of the shop by a counter with a glass screen. It is open from nine until half past five on "
        "weekdays and until midday on Saturdays. Most of the customers are local, and the woman who runs "
        "it knows many of them by name. In the morning people come in to post parcels, buy stamps and "
        "collect pensions, and there is often a short queue that moves slowly because everyone stops to "
        "talk. The scales on the counter weigh each parcel, and a small printer produces the label with "
        "the postage and a barcode for tracking. Larger items are stored in a cage at the back until "
        "the van collects them at four o'clock. Several times a week someone asks for help filling in a "
        "form, renewing a vehicle licence or sending money abroad, and the process is explained "
        "patiently, sometimes more than once. The noticeboard beside the door is covered with cards "
        "advertising a lost cat, piano lessons, a room to let and a lawnmower for sale. In December the "
        "post office becomes much busier. Extra staff are hired for the month, the queue stretches out "
        "of the door and the floor behind the counter is covered with sacks of cards and parcels. By "
        "the second week of January everything returns to normal, and the woman behind the counter "
        "takes her own holiday, leaving a neighbour to keep things running.")),
    NeutralText("s6_sportsday", (
        "Sports day at the primary school takes place on the last Friday before the summer holidays. "
        "Preparations begin a week earlier, when the caretaker marks out lanes on the playing field "
        "with white paint and the teachers sort the children into four coloured teams. On the morning "
        "itself, parents arrive with folding chairs and arrange themselves along the edge of the field, "
        "and the head teacher makes a short speech through a slightly unreliable microphone. The events "
        "are deliberately simple. There are races for each year group, a relay, an egg-and-spoon race "
        "and a sack race that always produces more laughter than any other event. The youngest children "
        "set off with enthusiasm in every direction except the finish line, and a teacher waits at the "
        "end to guide them across. Points are recorded on a large board beside the staff table, and the "
        "totals are updated after each event. The older pupils take the competition more seriously, "
        "practising their starts and arguing about whether someone set off too early. At lunchtime the "
        "parents share picnics on the grass while the children compare their stickers. In the afternoon "
        "there is a parents' race, which a small number of parents take part in and a much larger "
        "number watch. The day ends with the announcement of the winning team, a round of applause and "
        "a long queue at the gate as everyone heads home tired and slightly sunburnt.")),
    NeutralText("s6_timetable", (
        "Twice a year the railway company changes its timetable, usually in May and December. For most "
        "passengers the change is barely noticeable, as trains leave a few minutes earlier or later than "
        "before. For others it is more significant, because a service they rely on is withdrawn or a "
        "connection is no longer possible. In the weeks before the change, posters appear at every "
        "station and the new times are published online, along with a summary of the main differences. "
        "Staff at the ticket office answer the same questions many times a day, and the information "
        "screens show reminders in bright yellow text. Behind the scenes, the planning begins much "
        "earlier. Each train needs a driver and a guard, a set of carriages and a path along the track "
        "that does not conflict with any other service. When one train is moved, several others may "
        "have to move as well, so the planners work through the changes carefully, checking each "
        "station and junction in turn. Freight trains, engineering work and the times at which "
        "platforms are free all have to be taken into account. Once the new timetable starts, the first "
        "few days are watched closely. Small problems, such as a connection that is too tight or a "
        "train that is regularly crowded, are noted and sometimes fixed in a later update. After a few "
        "weeks most regular passengers have adjusted their routines, and the new times become simply "
        "the times the trains run.")),
)

# =====================================================================================
# Lookup and import-time checks
# =====================================================================================

SETS: dict[str, tuple[OpenPrompt, ...]] = {
    "search": SEARCH_PROMPTS,
    "heldout": HELDOUT_PROMPTS,
    "neutral": NEUTRAL_PROMPTS,
    "degeneracy": DEGENERACY_PROMPTS,
}


def texts(prompts: tuple[OpenPrompt, ...]) -> list[str]:
    """The prompt texts, in order (what `m5.prereg.prompt_set_hash` takes)."""
    return [p.text for p in prompts]


def prompt_set(split: str) -> tuple[OpenPrompt, ...]:
    try:
        return SETS[split]
    except KeyError:
        raise KeyError(f"unknown prompt set {split!r}; one of {sorted(SETS)}") from None


def find(prompt_id: str) -> tuple[str, OpenPrompt]:
    """(set name, prompt) for an id in any set; unknown ids raise."""
    for name, prompts in SETS.items():
        for p in prompts:
            if p.id == prompt_id:
                return name, p
    raise KeyError(f"unknown prompt id {prompt_id!r}")


def normalise(text: str) -> str:
    """Case, punctuation and spacing removed: two prompts that differ only there are one prompt."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'\-]+", text))


def _check() -> None:
    all_prompts = [p for s in SETS.values() for p in s]
    ids = [p.id for p in all_prompts] + [t.id for t in NEUTRAL_TEXTS]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate prompt id; ids key every row that will ever be joined")
    if len(SEARCH_PROMPTS) != 66:
        raise AssertionError("the search set is M3's 66 prompts")
    if len(HELDOUT_PROMPTS) < 64 or len(NEUTRAL_PROMPTS) < 16 or len(DEGENERACY_PROMPTS) < 8 \
            or len(NEUTRAL_TEXTS) < 8:
        raise AssertionError("a prompt set is below its planned size")
    for name, prompts in SETS.items():
        norm = [normalise(p.text) for p in prompts]
        if len(set(norm)) != len(norm):
            raise AssertionError(f"prompt set {name!r} contains duplicates")
    overlap = {normalise(p.text) for p in SEARCH_PROMPTS} & {normalise(p.text) for p in HELDOUT_PROMPTS}
    if overlap:
        raise AssertionError(f"held-out prompts overlap the search set: {sorted(overlap)}")
    for t in NEUTRAL_TEXTS:
        if not 200 <= word_count(t.text) <= 300:
            raise AssertionError(f"neutral text {t.id} has {word_count(t.text)} words; 200-300 expected")
    check_texts([p.text for p in all_prompts] + [t.text for t in NEUTRAL_TEXTS] + [S6_REQUEST],
                what="influence prompt")


_check()
